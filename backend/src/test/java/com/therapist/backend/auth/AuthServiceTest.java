package com.therapist.backend.auth;

import org.junit.jupiter.api.Test;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;

import java.util.Optional;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class AuthServiceTest {
    private static final class StubAuthEmailService extends AuthEmailService {
        private boolean fallbackVerificationSend;
        private int verificationSends;
        private int loginSends;
        private int resetSends;

        private StubAuthEmailService() {
            super(mock(JavaMailSender.class));
        }

        @Override
        public CodeDeliveryResult dispatchVerificationCode(String to, String code) {
            verificationSends++;
            if (fallbackVerificationSend) {
                return CodeDeliveryResult.fallback(code, "smtp down");
            }
            return CodeDeliveryResult.success();
        }

        @Override
        public CodeDeliveryResult dispatchLoginCode(String to, String code) {
            loginSends++;
            return CodeDeliveryResult.success();
        }

        @Override
        public CodeDeliveryResult dispatchPasswordResetCode(String to, String code) {
            resetSends++;
            return CodeDeliveryResult.success();
        }
    }

    @Test
    void startLogin_normalizesLegacyUserAndReturnsDirectLoginWhenRemembered() {
        UserRepository userRepository = mock(UserRepository.class);
        StubAuthEmailService authEmailService = new StubAuthEmailService();
        AuthService authService = new AuthService(userRepository, authEmailService);

        UserEntity user = new UserEntity();
        user.setId(UUID.randomUUID().toString());
        user.setEmail("legacy@example.com");
        user.setUsername(null);
        user.setSex(null);
        user.setAge(null);
        user.setPasswordHash(new BCryptPasswordEncoder().encode("secret123"));
        user.setVerified(null);
        user.setCreatedAt(null);
        user.setEmailCodeHash(null);

        when(userRepository.findByEmail("legacy@example.com")).thenReturn(Optional.of(user));
        when(userRepository.save(any(UserEntity.class))).thenAnswer(invocation -> invocation.getArgument(0));

        AuthService.StartLoginResult result = authService.startLogin(
                "legacy@example.com",
                "secret123",
                true
        );

        assertFalse(result.requires2fa());
        assertNotNull(result.loginResult());
        assertNotNull(result.loginResult().accessToken());
        assertEquals("legacy@example.com", result.loginResult().user().email());
        assertNotNull(result.loginResult().user().username());
        assertEquals("unknown", result.loginResult().user().sex());
        assertEquals(0, result.loginResult().user().age());
        assertTrue(result.loginResult().user().is_verified());
        verify(userRepository, atLeastOnce()).save(any(UserEntity.class));
        assertEquals(0, authEmailService.verificationSends);
        assertEquals(0, authEmailService.loginSends);
    }

    @Test
    void startLogin_acceptsLegacyPlaintextPasswordAndUpgradesHash() {
        UserRepository userRepository = mock(UserRepository.class);
        StubAuthEmailService authEmailService = new StubAuthEmailService();
        AuthService authService = new AuthService(userRepository, authEmailService);

        UserEntity user = new UserEntity();
        user.setId(UUID.randomUUID().toString());
        user.setEmail("legacy-plain@example.com");
        user.setUsername("legacyplain");
        user.setSex("M");
        user.setAge(31);
        user.setPasswordHash("secret123");
        user.setVerified(true);
        user.setCreatedAt(java.time.Instant.now());

        when(userRepository.findByEmail("legacy-plain@example.com")).thenReturn(Optional.of(user));
        when(userRepository.save(any(UserEntity.class))).thenAnswer(invocation -> invocation.getArgument(0));

        AuthService.StartLoginResult result = authService.startLogin(
                "legacy-plain@example.com",
                "secret123",
                true
        );

        assertFalse(result.requires2fa());
        assertNotNull(result.loginResult());
        assertNotNull(result.loginResult().accessToken());
        assertTrue(new BCryptPasswordEncoder().matches("secret123", user.getPasswordHash()));
        verify(userRepository, atLeastOnce()).save(any(UserEntity.class));
    }

    @Test
    void startLogin_rejectsMalformedPasswordHashWithoutThrowingServerError() {
        UserRepository userRepository = mock(UserRepository.class);
        StubAuthEmailService authEmailService = new StubAuthEmailService();
        AuthService authService = new AuthService(userRepository, authEmailService);

        UserEntity user = new UserEntity();
        user.setId(UUID.randomUUID().toString());
        user.setEmail("broken@example.com");
        user.setUsername("broken");
        user.setSex("F");
        user.setAge(29);
        user.setPasswordHash("$2a$not-a-valid-bcrypt-hash");
        user.setVerified(true);
        user.setCreatedAt(java.time.Instant.now());

        when(userRepository.findByEmail("broken@example.com")).thenReturn(Optional.of(user));

        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                () -> authService.startLogin("broken@example.com", "secret123", true)
        );

        assertEquals("Identifiants invalides.", error.getMessage());
    }

    @Test
    void register_returnsFallbackCodeWhenVerificationEmailFallsBack() {
        UserRepository userRepository = mock(UserRepository.class);
        StubAuthEmailService authEmailService = new StubAuthEmailService();
        authEmailService.fallbackVerificationSend = true;
        AuthService authService = new AuthService(userRepository, authEmailService);

        when(userRepository.findByEmail("new@example.com")).thenReturn(Optional.empty());
        when(userRepository.findByUsername("newuser")).thenReturn(Optional.empty());
        when(userRepository.save(any(UserEntity.class))).thenAnswer(invocation -> invocation.getArgument(0));

        AuthService.RegisterResult result = authService.register(
                "new@example.com",
                "newuser",
                "F",
                24,
                "secret123",
                "secret123"
        );

        assertTrue(result.delivery().usedFallback());
        assertEquals("new@example.com", result.user().email());
        assertEquals(1, authEmailService.verificationSends);
        verify(userRepository, atLeastOnce()).save(any(UserEntity.class));
    }

    @Test
    void resendVerificationEmail_refreshesCodeForUnverifiedUser() {
        UserRepository userRepository = mock(UserRepository.class);
        StubAuthEmailService authEmailService = new StubAuthEmailService();
        AuthService authService = new AuthService(userRepository, authEmailService);

        UserEntity user = new UserEntity();
        user.setId(UUID.randomUUID().toString());
        user.setEmail("pending@example.com");
        user.setUsername("pending");
        user.setSex("F");
        user.setAge(23);
        user.setPasswordHash(new BCryptPasswordEncoder().encode("secret123"));
        user.setVerified(false);

        when(userRepository.findByEmail("pending@example.com")).thenReturn(Optional.of(user));
        when(userRepository.save(any(UserEntity.class))).thenAnswer(invocation -> invocation.getArgument(0));

        authService.resendVerificationEmail("pending@example.com");

        assertNotNull(user.getEmailCodeHash());
        assertNotNull(user.getEmailCodeExpiresAt());
        assertEquals(1, authEmailService.verificationSends);
    }
}
