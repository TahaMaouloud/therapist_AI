package com.therapist.backend.controller;

import com.therapist.backend.auth.AuthDtos;
import com.therapist.backend.auth.AuthEmailService;
import com.therapist.backend.auth.AuthService;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

@RestController
@RequestMapping("/auth")
public class AuthController {
    private final AuthService authService;

    public AuthController(AuthService authService) {
        this.authService = authService;
    }

    @PostMapping("/register")
    public ResponseEntity<?> register(
            @RequestParam String email,
            @RequestParam String username,
            @RequestParam String sex,
            @RequestParam int age,
            @RequestParam String password,
            @RequestParam("confirm_password") String confirmPassword
    ) {
        try {
            AuthService.RegisterResult result = authService.register(
                    email, username, sex, age, password, confirmPassword
            );
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("message", deliveryMessage(
                    "Compte cree. Un code a ete envoye par email.",
                    "Compte cree. SMTP indisponible en local, le code de verification est affiche dans l'application.",
                    result.delivery()
            ));
            body.put("user", result.user());
            appendDelivery(body, result.delivery());
            return ResponseEntity.ok(body);
        } catch (IllegalArgumentException ex) {
            return ResponseEntity.badRequest().body(Map.of("detail", ex.getMessage()));
        } catch (IllegalStateException ex) {
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("detail", ex.getMessage()));
        } catch (Exception ex) {
            return internalError(ex);
        }
    }

    @PostMapping("/verify-email")
    public ResponseEntity<?> verify(@Valid @RequestBody AuthDtos.VerifyEmailRequest req) {
        try {
            AuthDtos.PublicUser user = authService.verifyEmail(req.email(), req.code());
            return ResponseEntity.ok(Map.of("message", "Email verifie.", "user", user));
        } catch (IllegalArgumentException ex) {
            return ResponseEntity.badRequest().body(Map.of("detail", ex.getMessage()));
        } catch (Exception ex) {
            return internalError(ex);
        }
    }

    @PostMapping("/resend-verification")
    public ResponseEntity<?> resendVerification(@Valid @RequestBody AuthDtos.ResendVerificationRequest req) {
        try {
            AuthEmailService.CodeDeliveryResult delivery = authService.resendVerificationEmail(req.email());
            Map<String, Object> body = new LinkedHashMap<>();
            body.put(
                    "message",
                    deliveryMessage(
                            "Un nouveau code de verification a ete envoye. Verifie aussi le dossier spam.",
                            "SMTP indisponible en local, le nouveau code de verification est affiche dans l'application.",
                            delivery
                    )
            );
            appendDelivery(body, delivery);
            return ResponseEntity.ok(body);
        } catch (IllegalArgumentException ex) {
            return ResponseEntity.badRequest().body(Map.of("detail", ex.getMessage()));
        } catch (IllegalStateException ex) {
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("detail", ex.getMessage()));
        } catch (Exception ex) {
            return internalError(ex);
        }
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@Valid @RequestBody AuthDtos.LoginRequest req) {
        if (!req.human_check()) {
            return ResponseEntity.badRequest().body(Map.of(
                    "detail", "Veuillez cocher \"Je ne suis pas un robot\"."
            ));
        }
        try {
            AuthService.StartLoginResult result = authService.startLogin(
                    req.email(),
                    req.password(),
                    req.remember_me()
            );
            if (result.requires2fa()) {
                Map<String, Object> body = new LinkedHashMap<>();
                body.put("requires_2fa", true);
                body.put(
                        "message",
                        deliveryMessage(
                                "Code de connexion envoye par email.",
                                "SMTP indisponible en local, le code de connexion est affiche dans l'application.",
                                result.delivery()
                        )
                );
                appendDelivery(body, result.delivery());
                return ResponseEntity.ok(body);
            }

            AuthService.LoginResult loginResult = result.loginResult();
            return ResponseEntity.ok(Map.of(
                    "requires_2fa", false,
                    "access_token", loginResult.accessToken(),
                    "user", loginResult.user(),
                    "message", "Connexion reussie."
            ));
        } catch (IllegalArgumentException ex) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("detail", ex.getMessage()));
        } catch (IllegalStateException ex) {
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("detail", ex.getMessage()));
        } catch (Exception ex) {
            return internalError(ex);
        }
    }

    @PostMapping("/login/verify-code")
    public ResponseEntity<?> verifyLoginCode(@Valid @RequestBody AuthDtos.LoginCodeRequest req) {
        try {
            AuthService.LoginResult result = authService.verifyLoginCode(req.email(), req.code());
            return ResponseEntity.ok(Map.of(
                    "access_token", result.accessToken(),
                    "user", result.user()
            ));
        } catch (IllegalArgumentException ex) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("detail", ex.getMessage()));
        } catch (Exception ex) {
            return internalError(ex);
        }
    }

    @PostMapping("/forgot-password")
    public ResponseEntity<?> forgotPassword(@Valid @RequestBody AuthDtos.ForgotPasswordRequest req) {
        try {
            AuthEmailService.CodeDeliveryResult delivery = authService.forgotPassword(req.email());
            Map<String, Object> body = new LinkedHashMap<>();
            body.put(
                    "message",
                    deliveryMessage(
                            "Si un compte existe avec cette adresse email, un code de reinitialisation a ete envoye.",
                            "Si un compte existe avec cette adresse email, SMTP est indisponible en local et le code de reinitialisation est affiche dans l'application.",
                            delivery
                    )
            );
            appendDelivery(body, delivery);
            return ResponseEntity.ok(body);
        } catch (IllegalArgumentException ex) {
            return ResponseEntity.badRequest().body(Map.of("detail", ex.getMessage()));
        } catch (IllegalStateException ex) {
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("detail", ex.getMessage()));
        } catch (Exception ex) {
            return internalError(ex);
        }
    }

    @PostMapping("/reset-password")
    public ResponseEntity<?> resetPassword(@Valid @RequestBody AuthDtos.ResetPasswordRequest req) {
        try {
            authService.resetPassword(req.email(), req.code(), req.newPassword(), req.confirmPassword());
            return ResponseEntity.ok(Map.of(
                    "message", "Mot de passe reinitialise avec succes. Vous pouvez maintenant vous connecter."
            ));
        } catch (IllegalArgumentException ex) {
            return ResponseEntity.badRequest().body(Map.of("detail", ex.getMessage()));
        } catch (Exception ex) {
            return internalError(ex);
        }
    }

    @GetMapping("/me")
    public ResponseEntity<?> me(@RequestHeader(value = "Authorization", required = false) String authHeader) {
        String token = extractBearer(authHeader);
        if (token == null) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("detail", "Authorization manquant."));
        }
        try {
            return ResponseEntity.ok(Map.of("user", authService.me(token)));
        } catch (IllegalArgumentException ex) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("detail", ex.getMessage()));
        } catch (Exception ex) {
            return internalError(ex);
        }
    }

    @PatchMapping("/profile")
    public ResponseEntity<?> updateProfile(
            @RequestHeader(value = "Authorization", required = false) String authHeader,
            @Valid @RequestBody AuthDtos.UpdateProfileRequest req
    ) {
        String token = extractBearer(authHeader);
        if (token == null) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("detail", "Authorization manquant."));
        }
        try {
            AuthDtos.PublicUser user = authService.updateUsername(token, req.username());
            return ResponseEntity.ok(Map.of("message", "Profil mis a jour.", "user", user));
        } catch (IllegalArgumentException ex) {
            return ResponseEntity.badRequest().body(Map.of("detail", ex.getMessage()));
        } catch (Exception ex) {
            return internalError(ex);
        }
    }

    @PostMapping("/change-password")
    public ResponseEntity<?> changePassword(
            @RequestHeader(value = "Authorization", required = false) String authHeader,
            @Valid @RequestBody AuthDtos.ChangePasswordRequest req
    ) {
        String token = extractBearer(authHeader);
        if (token == null) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("detail", "Authorization manquant."));
        }
        try {
            authService.changePassword(token, req.oldPassword(), req.newPassword());
            return ResponseEntity.ok(Map.of("message", "Mot de passe mis a jour."));
        } catch (IllegalArgumentException ex) {
            return ResponseEntity.badRequest().body(Map.of("detail", ex.getMessage()));
        } catch (Exception ex) {
            return internalError(ex);
        }
    }

    @PostMapping(value = "/profile-photo", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<?> updateProfilePhoto(
            @RequestHeader(value = "Authorization", required = false) String authHeader,
            @RequestPart("photo") MultipartFile photo
    ) {
        String token = extractBearer(authHeader);
        if (token == null) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("detail", "Authorization manquant."));
        }
        try {
            String dataUrl = buildPhotoDataUrl(photo);
            AuthDtos.PublicUser user = authService.updateProfilePhoto(token, dataUrl);
            return ResponseEntity.ok(Map.of("message", "Photo de profil mise a jour.", "user", user));
        } catch (IllegalArgumentException ex) {
            return ResponseEntity.badRequest().body(Map.of("detail", ex.getMessage()));
        } catch (Exception ex) {
            return ResponseEntity.internalServerError().body(Map.of("detail", ex.getMessage()));
        }
    }

    @DeleteMapping("/profile-photo")
    public ResponseEntity<?> deleteProfilePhoto(
            @RequestHeader(value = "Authorization", required = false) String authHeader
    ) {
        String token = extractBearer(authHeader);
        if (token == null) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("detail", "Authorization manquant."));
        }
        try {
            AuthDtos.PublicUser user = authService.clearProfilePhoto(token);
            return ResponseEntity.ok(Map.of("message", "Photo supprimee.", "user", user));
        } catch (IllegalArgumentException ex) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("detail", ex.getMessage()));
        } catch (Exception ex) {
            return internalError(ex);
        }
    }

    @DeleteMapping("/account")
    public ResponseEntity<?> deleteAccount(
            @RequestHeader(value = "Authorization", required = false) String authHeader
    ) {
        String token = extractBearer(authHeader);
        if (token == null) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("detail", "Authorization manquant."));
        }
        try {
            authService.deleteAccount(token);
            return ResponseEntity.ok(Map.of("message", "Compte supprime."));
        } catch (IllegalArgumentException ex) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("detail", ex.getMessage()));
        } catch (Exception ex) {
            return internalError(ex);
        }
    }

    private String extractBearer(String authHeader) {
        if (authHeader == null || !authHeader.toLowerCase().startsWith("bearer ")) {
            return null;
        }
        return authHeader.substring(7).trim();
    }

    private String buildPhotoDataUrl(MultipartFile photo) throws Exception {
        if (photo == null || photo.isEmpty()) {
            throw new IllegalArgumentException("Aucune photo envoyee.");
        }
        if (photo.getSize() > 2_500_000) {
            throw new IllegalArgumentException("Image trop lourde. Maximum 2.5 MB.");
        }
        String contentType = photo.getContentType() == null
                ? ""
                : photo.getContentType().toLowerCase(Locale.ROOT);
        if (!contentType.startsWith("image/")) {
            throw new IllegalArgumentException("Format image invalide.");
        }
        byte[] bytes = photo.getBytes();
        String base64 = Base64.getEncoder().encodeToString(bytes);
        return "data:" + contentType + ";base64," + base64;
    }

    private ResponseEntity<?> internalError(Exception ex) {
        return ResponseEntity.internalServerError().body(Map.of("detail", extractErrorMessage(ex)));
    }

    private String extractErrorMessage(Throwable ex) {
        Throwable current = ex;
        while (current != null) {
            String message = current.getMessage();
            if (message != null && !message.isBlank()) {
                return message;
            }
            current = current.getCause();
        }
        return "Erreur interne du backend auth.";
    }

    private String deliveryMessage(
            String successMessage,
            String fallbackMessage,
            AuthEmailService.CodeDeliveryResult delivery
    ) {
        return delivery != null && delivery.usedFallback() ? fallbackMessage : successMessage;
    }

    private void appendDelivery(Map<String, Object> body, AuthEmailService.CodeDeliveryResult delivery) {
        if (body == null || delivery == null) {
            return;
        }
        if (delivery.usedFallback()) {
            body.put("delivery_mode", "fallback");
            body.put("fallback_code", delivery.fallbackCode());
            body.put("delivery_detail", delivery.detailMessage());
            return;
        }
        if (delivery.delivered()) {
            body.put("delivery_mode", "email");
        }
    }
}
