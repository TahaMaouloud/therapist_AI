package com.therapist.backend.auth;

import org.springframework.stereotype.Service;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;

import java.time.Instant;
import java.util.Base64;
import java.util.Locale;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;

@Service
public class AuthService {
    private final UserRepository userRepository;
    private final AuthEmailService authEmailService;
    private final BCryptPasswordEncoder passwordEncoder = new BCryptPasswordEncoder();
    private final Map<String, String> sessions = new ConcurrentHashMap<>();
    private final Random random = new Random();
    private static final int CODE_EXPIRY_MINUTES = 10;

    public AuthService(UserRepository userRepository, AuthEmailService authEmailService) {
        this.userRepository = userRepository;
        this.authEmailService = authEmailService;
    }

    public synchronized RegisterResult register(
            String email,
            String username,
            String sex,
            int age,
            String password,
            String confirmPassword
    ) {
        String normalizedEmail = email.trim().toLowerCase(Locale.ROOT);
        String normalizedUsername = username.trim().toLowerCase(Locale.ROOT);

        if (!password.equals(confirmPassword)) {
            throw new IllegalArgumentException("Passwords differents.");
        }
        if (userRepository.findByEmail(normalizedEmail).isPresent()) {
            throw new IllegalArgumentException("Email deja utilise.");
        }
        if (userRepository.findByUsername(normalizedUsername).isPresent()) {
            throw new IllegalArgumentException("Username deja utilise.");
        }

        String emailCode = generateSixDigitCode();
        UserEntity user = new UserEntity();
        user.setId(UUID.randomUUID().toString());
        user.setEmail(normalizedEmail);
        user.setUsername(normalizedUsername);
        user.setSex(sex);
        user.setAge(age);
        user.setPasswordHash(passwordEncoder.encode(password));
        user.setPhotoPath(null);
        user.setVerified(false);
        assignFreshEmailCode(user, emailCode);
        user.setCreatedAt(Instant.now());
        userRepository.save(user);
        AuthEmailService.CodeDeliveryResult delivery;
        try {
            delivery = authEmailService.dispatchVerificationCode(
                    user.getEmail(),
                    emailCode
            );
        } catch (RuntimeException ex) {
            safeDeleteUser(user.getId());
            throw ex;
        }
        return new RegisterResult(publicUser(user), delivery);
    }

    public synchronized AuthEmailService.CodeDeliveryResult resendVerificationEmail(String email) {
        String normalizedEmail = email.trim().toLowerCase(Locale.ROOT);
        UserEntity user = userRepository.findByEmail(normalizedEmail).orElse(null);
        if (user == null) {
            throw new IllegalArgumentException("Compte introuvable.");
        }
        if (user.isVerified()) {
            throw new IllegalArgumentException("Email deja verifie.");
        }

        String emailCode = generateSixDigitCode();
        assignFreshEmailCode(user, emailCode);
        normalizeLegacyUserState(user);
        userRepository.save(user);
        return authEmailService.dispatchVerificationCode(user.getEmail(), emailCode);
    }

    public synchronized AuthDtos.PublicUser verifyEmail(String email, String code) {
        String normalizedEmail = email.trim().toLowerCase(Locale.ROOT);
        UserEntity user = userRepository.findByEmail(normalizedEmail).orElse(null);
        if (user == null) {
            throw new IllegalArgumentException("Compte introuvable.");
        }
        if (user.getEmailCodeHash() == null || user.getEmailCodeExpiresAt() == null) {
            throw new IllegalArgumentException("Aucun code de verification actif.");
        }
        if (Instant.now().isAfter(user.getEmailCodeExpiresAt())) {
            throw new IllegalArgumentException("Code de verification expire.");
        }
        if (!matchesStoredSecret(code, user.getEmailCodeHash())) {
            throw new IllegalArgumentException("Code de verification invalide.");
        }
        user.setVerified(true);
        user.setEmailCodeHash(null);
        user.setEmailCodeExpiresAt(null);
        normalizeLegacyUserState(user);
        userRepository.save(user);
        return publicUser(user);
    }

    public synchronized StartLoginResult startLogin(String email, String password, boolean rememberMe) {
        String normalizedEmail = email.trim().toLowerCase(Locale.ROOT);
        UserEntity user = userRepository.findByEmail(normalizedEmail).orElse(null);
        if (user == null || !matchesStoredSecret(password, user.getPasswordHash())) {
            throw new IllegalArgumentException("Identifiants invalides.");
        }
        boolean changed = normalizeLegacyUserState(user);
        if (upgradeLegacyPasswordHashIfNeeded(user, password)) {
            changed = true;
        }
        if (!user.isVerified()) {
            throw new IllegalArgumentException("Email non verifie.");
        }
        if (rememberMe) {
            user.setLoginCodeHash(null);
            user.setLoginCodeExpiresAt(null);
            userRepository.save(user);
            return new StartLoginResult(
                    false,
                    new LoginResult(createSessionToken(user), publicUser(user)),
                    null
            );
        }

        String loginCode = generateSixDigitCode();
        user.setLoginCodeHash(passwordEncoder.encode(loginCode));
        user.setLoginCodeExpiresAt(Instant.now().plusSeconds(CODE_EXPIRY_MINUTES * 60L));
        if (changed || user.getLoginCodeHash() != null) {
            userRepository.save(user);
        }
        AuthEmailService.CodeDeliveryResult delivery = authEmailService.dispatchLoginCode(
                user.getEmail(),
                loginCode
        );
        return new StartLoginResult(true, null, delivery);
    }

    public synchronized LoginResult verifyLoginCode(String email, String code) {
        String normalizedEmail = email.trim().toLowerCase(Locale.ROOT);
        UserEntity user = userRepository.findByEmail(normalizedEmail).orElse(null);
        if (user == null) {
            throw new IllegalArgumentException("Compte introuvable.");
        }
        if (user.getLoginCodeHash() == null || user.getLoginCodeExpiresAt() == null) {
            throw new IllegalArgumentException("Aucun code de connexion actif.");
        }
        if (Instant.now().isAfter(user.getLoginCodeExpiresAt())) {
            throw new IllegalArgumentException("Code de connexion expire.");
        }
        if (!matchesStoredSecret(code, user.getLoginCodeHash())) {
            throw new IllegalArgumentException("Code de connexion invalide.");
        }
        user.setLoginCodeHash(null);
        user.setLoginCodeExpiresAt(null);
        normalizeLegacyUserState(user);
        userRepository.save(user);

        return new LoginResult(createSessionToken(user), publicUser(user));
    }

    public synchronized AuthEmailService.CodeDeliveryResult forgotPassword(String email) {
        String normalizedEmail = email.trim().toLowerCase(Locale.ROOT);
        UserEntity user = userRepository.findByEmail(normalizedEmail).orElse(null);
        if (user == null) {
            // Don't reveal if email exists or not for security
            return AuthEmailService.CodeDeliveryResult.success();
        }

        String resetCode = generateSixDigitCode();
        user.setResetCodeHash(passwordEncoder.encode(resetCode));
        user.setResetCodeExpiresAt(Instant.now().plusSeconds(CODE_EXPIRY_MINUTES * 60L));
        normalizeLegacyUserState(user);
        userRepository.save(user);
        return authEmailService.dispatchPasswordResetCode(user.getEmail(), resetCode);
    }

    public synchronized void resetPassword(String email, String code, String newPassword, String confirmPassword) {
        String normalizedEmail = email.trim().toLowerCase(Locale.ROOT);
        UserEntity user = userRepository.findByEmail(normalizedEmail).orElse(null);
        if (user == null) {
            throw new IllegalArgumentException("Compte introuvable.");
        }

        if (!newPassword.equals(confirmPassword)) {
            throw new IllegalArgumentException("Les mots de passe ne correspondent pas.");
        }

        if (newPassword.length() < 6) {
            throw new IllegalArgumentException("Le mot de passe doit contenir au moins 6 caracteres.");
        }

        if (user.getResetCodeHash() == null || user.getResetCodeExpiresAt() == null) {
            throw new IllegalArgumentException("Aucun code de reinitialisation actif.");
        }
        if (Instant.now().isAfter(user.getResetCodeExpiresAt())) {
            throw new IllegalArgumentException("Code de reinitialisation expire.");
        }
        if (!matchesStoredSecret(code, user.getResetCodeHash())) {
            throw new IllegalArgumentException("Code de reinitialisation invalide.");
        }

        user.setPasswordHash(passwordEncoder.encode(newPassword));
        user.setResetCodeHash(null);
        user.setResetCodeExpiresAt(null);
        normalizeLegacyUserState(user);
        userRepository.save(user);
    }

    public synchronized AuthDtos.PublicUser updateUsername(String bearerToken, String username) {
        UserEntity user = resolveSessionUser(bearerToken);
        String normalizedUsername = username == null
                ? ""
                : username.trim().toLowerCase(Locale.ROOT);

        if (normalizedUsername.isBlank()) {
            throw new IllegalArgumentException("Username obligatoire.");
        }
        if (normalizedUsername.length() < 3 || normalizedUsername.length() > 32) {
            throw new IllegalArgumentException("Le username doit contenir entre 3 et 32 caracteres.");
        }
        if (!normalizedUsername.matches("[a-z0-9._-]+")) {
            throw new IllegalArgumentException("Username invalide. Utilisez lettres, chiffres, '.', '_' ou '-'.");
        }

        UserEntity existing = userRepository.findByUsername(normalizedUsername).orElse(null);
        if (existing != null && !existing.getId().equals(user.getId())) {
            throw new IllegalArgumentException("Username deja utilise.");
        }

        user.setUsername(normalizedUsername);
        normalizeLegacyUserState(user);
        userRepository.save(user);
        return publicUser(user);
    }

    public synchronized void changePassword(String bearerToken, String oldPassword, String newPassword) {
        UserEntity user = resolveSessionUser(bearerToken);
        if (oldPassword == null || newPassword == null) {
            throw new IllegalArgumentException("Ancien et nouveau mot de passe obligatoires.");
        }
        if (!matchesStoredSecret(oldPassword, user.getPasswordHash())) {
            throw new IllegalArgumentException("Ancien mot de passe incorrect.");
        }
        if (newPassword.length() < 6) {
            throw new IllegalArgumentException("Le nouveau mot de passe doit contenir au moins 6 caracteres.");
        }
        if (matchesStoredSecret(newPassword, user.getPasswordHash())) {
            throw new IllegalArgumentException("Le nouveau mot de passe doit etre different.");
        }

        user.setPasswordHash(passwordEncoder.encode(newPassword));
        user.setLoginCodeHash(null);
        user.setLoginCodeExpiresAt(null);
        user.setResetCodeHash(null);
        user.setResetCodeExpiresAt(null);
        normalizeLegacyUserState(user);
        userRepository.save(user);
    }

    public AuthDtos.PublicUser me(String bearerToken) {
        return publicUser(resolveSessionUser(bearerToken));
    }

    public AuthDtos.PublicUser updateProfilePhoto(String bearerToken, String photoDataUrl) {
        UserEntity user = resolveSessionUser(bearerToken);
        user.setPhotoPath(photoDataUrl);
        normalizeLegacyUserState(user);
        userRepository.save(user);
        return publicUser(user);
    }

    public AuthDtos.PublicUser clearProfilePhoto(String bearerToken) {
        UserEntity user = resolveSessionUser(bearerToken);
        user.setPhotoPath(null);
        normalizeLegacyUserState(user);
        userRepository.save(user);
        return publicUser(user);
    }
    public synchronized void deleteAccount(String bearerToken) {
        UserEntity user = resolveSessionUser(bearerToken);
        String userId = user.getId();
        userRepository.delete(user);
        sessions.entrySet().removeIf(entry -> userId.equals(entry.getValue()));
    }
    public String resolveUserId(String bearerToken) {
        return sessions.get(bearerToken);
    }

    private UserEntity resolveSessionUser(String bearerToken) {
        String userId = sessions.get(bearerToken);
        if (userId == null) {
            throw new IllegalArgumentException("Session invalide.");
        }
        UserEntity user = userRepository.findById(userId).orElse(null);
        if (user == null) {
            throw new IllegalArgumentException("Utilisateur introuvable.");
        }
        return user;
    }

    private AuthDtos.PublicUser publicUser(UserEntity user) {
        String username = user.getUsername();
        if (normalizeLegacyUserState(user)) {
            userRepository.save(user);
        }
        Instant createdAt = user.getCreatedAt();
        int age = user.getAge() == null ? 0 : user.getAge();
        boolean verified = user.isVerified();
        if (username == null || username.isBlank()) {
            username = fallbackUsername(user);
        }
        return new AuthDtos.PublicUser(
                user.getId(),
                user.getEmail(),
                username,
                user.getSex(),
                age,
                user.getPhotoPath(),
                verified,
                createdAt.toString()
        );
    }

    private boolean normalizeLegacyUserState(UserEntity user) {
        boolean changed = false;

        if (user.getCreatedAt() == null) {
            user.setCreatedAt(Instant.now());
            changed = true;
        }
        if (user.getAge() == null) {
            user.setAge(0);
            changed = true;
        }
        if (user.getVerified() == null) {
            user.setVerified(user.getEmailCodeHash() == null || user.getEmailCodeHash().isBlank());
            changed = true;
        }
        if (isBlank(user.getSex())) {
            user.setSex("unknown");
            changed = true;
        }
        if (isBlank(user.getUsername())) {
            user.setUsername(fallbackUsername(user));
            changed = true;
        }

        return changed;
    }

    private String fallbackUsername(UserEntity user) {
        String email = user.getEmail() == null ? "" : user.getEmail().trim().toLowerCase(Locale.ROOT);
        String localPart = email.contains("@") ? email.substring(0, email.indexOf('@')) : email;
        String cleaned = localPart.replaceAll("[^a-z0-9._-]", "");
        if (cleaned.isBlank()) {
            cleaned = "user";
        }

        String suffix = user.getId() == null ? UUID.randomUUID().toString() : user.getId();
        suffix = suffix.replaceAll("[^a-zA-Z0-9]", "");
        if (suffix.length() > 6) {
            suffix = suffix.substring(Math.max(0, suffix.length() - 6));
        }
        return (cleaned + "_" + suffix).toLowerCase(Locale.ROOT);
    }

    private boolean isBlank(String value) {
        return value == null || value.isBlank();
    }

    private void assignFreshEmailCode(UserEntity user, String emailCode) {
        user.setEmailCodeHash(passwordEncoder.encode(emailCode));
        user.setEmailCodeExpiresAt(Instant.now().plusSeconds(CODE_EXPIRY_MINUTES * 60L));
        user.setVerified(false);
    }

    private void safeDeleteUser(String userId) {
        if (userId == null || userId.isBlank()) {
            return;
        }
        try {
            userRepository.deleteById(userId);
        } catch (Exception ignored) {
            // Best effort rollback so a failed registration can be retried cleanly.
        }
    }

    private String generateSixDigitCode() {
        int value = 100000 + random.nextInt(900000);
        return Integer.toString(value);
    }

    private boolean matchesStoredSecret(String rawValue, String storedValue) {
        if (isBlank(rawValue) || isBlank(storedValue)) {
            return false;
        }
        String normalizedStoredValue = storedValue.trim();
        if (looksLikeBcryptHash(normalizedStoredValue)) {
            try {
                return passwordEncoder.matches(rawValue, normalizedStoredValue);
            } catch (IllegalArgumentException ex) {
                return false;
            }
        }
        return normalizedStoredValue.equals(rawValue);
    }

    private boolean upgradeLegacyPasswordHashIfNeeded(UserEntity user, String rawPassword) {
        if (user == null || isBlank(rawPassword)) {
            return false;
        }
        String storedPasswordHash = user.getPasswordHash();
        if (isBlank(storedPasswordHash)) {
            return false;
        }
        String normalizedStoredPasswordHash = storedPasswordHash.trim();
        if (looksLikeBcryptHash(normalizedStoredPasswordHash)) {
            return false;
        }
        if (!normalizedStoredPasswordHash.equals(rawPassword)) {
            return false;
        }
        user.setPasswordHash(passwordEncoder.encode(rawPassword));
        return true;
    }

    private boolean looksLikeBcryptHash(String value) {
        return value != null && value.matches("^\\$2[aby]?\\$\\d{2}\\$.{53}$");
    }

    private String createSessionToken(UserEntity user) {
        String token = Base64.getUrlEncoder()
                .withoutPadding()
                .encodeToString((UUID.randomUUID() + ":" + user.getId()).getBytes());
        sessions.put(token, user.getId());
        return token;
    }

    public record RegisterResult(AuthDtos.PublicUser user, AuthEmailService.CodeDeliveryResult delivery) {}
    public record StartLoginResult(
            boolean requires2fa,
            LoginResult loginResult,
            AuthEmailService.CodeDeliveryResult delivery
    ) {}
    public record LoginResult(String accessToken, AuthDtos.PublicUser user) {}
}
