package com.therapist.backend.auth;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;

public class AuthDtos {
    public record LoginRequest(
            @Email @NotBlank String email,
            @NotBlank String password,
            boolean remember_me,
            boolean human_check
    ) {}

    public record VerifyEmailRequest(
            @Email @NotBlank String email,
            @NotBlank String code
    ) {}

    public record ResendVerificationRequest(
            @Email @NotBlank String email
    ) {}

    public record LoginCodeRequest(
            @Email @NotBlank String email,
            @NotBlank String code
    ) {}

    public record ForgotPasswordRequest(
            @Email @NotBlank String email
    ) {}

    public record ResetPasswordRequest(
            @Email @NotBlank String email,
            @NotBlank String code,
            @NotBlank String newPassword,
            @NotBlank String confirmPassword
    ) {}

    public record UpdateProfileRequest(
            @NotBlank String username
    ) {}

    public record ChangePasswordRequest(
            @NotBlank String oldPassword,
            @NotBlank String newPassword
    ) {}

    public record PublicUser(
            String id,
            String email,
            String username,
            String sex,
            int age,
            String photo_path,
            boolean is_verified,
            String created_at
    ) {}
}
