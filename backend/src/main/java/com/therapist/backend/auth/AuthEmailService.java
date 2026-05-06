package com.therapist.backend.auth;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.mail.MailAuthenticationException;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Service;

@Service
public class AuthEmailService {
    public record CodeDeliveryResult(boolean delivered, String fallbackCode, String detailMessage) {
        public static CodeDeliveryResult success() {
            return new CodeDeliveryResult(true, null, "");
        }

        public static CodeDeliveryResult fallback(String fallbackCode, String detailMessage) {
            return new CodeDeliveryResult(false, fallbackCode, detailMessage);
        }

        public boolean usedFallback() {
            return !delivered && fallbackCode != null && !fallbackCode.isBlank();
        }
    }

    private final JavaMailSender mailSender;

    @Value("${MAIL_FROM:}")
    private String configuredFromAddress;

    @Value("${spring.mail.host:}")
    private String smtpHost;

    @Value("${spring.mail.username:}")
    private String smtpUsername;

    @Value("${spring.mail.password:}")
    private String smtpPassword;

    @Value("${AUTH_EMAIL_CONSOLE_FALLBACK:false}")
    private boolean consoleFallback;

    public AuthEmailService(JavaMailSender mailSender) {
        this.mailSender = mailSender;
    }

    public CodeDeliveryResult sendVerificationCode(String to, String code) {
        return send(
                to,
                "Therapist AI - Code de verification",
                "Ton code de verification est: " + code + "\n\n"
                        + "Ce code est valable pendant 10 minutes. Merci de ne pas le partager avec qui que ce soit.\n\n"
                        + "Si vous n'etes pas a l'origine de cette demande, vous pouvez simplement ignorer cet email.\n\n"
                        + "Cordialement,\n"
                        + "L'equipe Therapist AI",
                code
        );
    }

    public void sendLoginCode(String to, String code) {
        send(
                to,
                "Therapist AI - Code de connexion",
                "Bonjour,\n\n"
                        + "Pour finaliser votre connexion a l'application Therapist AI, veuillez utiliser le code de verification ci-dessous :\n\n"
                        + "Code de connexion : " + code + "\n\n"
                        + "Ce code est valable pendant 10 minutes. Merci de ne pas le partager avec qui que ce soit.\n\n"
                        + "Si vous n'etes pas a l'origine de cette demande, vous pouvez simplement ignorer cet email.\n\n"
                        + "Cordialement,\n"
                        + "L'equipe Therapist AI"
        );
    }

    public void sendPasswordResetCode(String to, String code) {
        send(
                to,
                "Therapist AI - Reinitialisation du mot de passe",
                "Bonjour,\n\n"
                        + "Vous avez demande la reinitialisation de votre mot de passe pour l'application Therapist AI.\n\n"
                        + "Code de reinitialisation : " + code + "\n\n"
                        + "Ce code est valable pendant 10 minutes. Si vous n'avez pas demande cette reinitialisation, vous pouvez simplement ignorer cet email.\n\n"
                        + "Cordialement,\n"
                        + "L'equipe Therapist AI"
        );
    }

    public CodeDeliveryResult dispatchVerificationCode(String to, String code) {
        return send(
                to,
                "Therapist AI - Code de verification",
                "Ton code de verification est: " + code + "\n\n"
                        + "Ce code est valable pendant 10 minutes. Merci de ne pas le partager.\n\n"
                        + "Si vous n'etes pas a l'origine de cette demande, ignorez simplement cet email.\n\n"
                        + "Cordialement,\n"
                        + "L'equipe Therapist AI",
                code
        );
    }

    public CodeDeliveryResult dispatchLoginCode(String to, String code) {
        return send(
                to,
                "Therapist AI - Code de connexion",
                "Bonjour,\n\n"
                        + "Pour finaliser votre connexion a Therapist AI, utilisez le code ci-dessous.\n\n"
                        + "Code de connexion : " + code + "\n\n"
                        + "Ce code est valable pendant 10 minutes. Merci de ne pas le partager.\n\n"
                        + "Si vous n'etes pas a l'origine de cette demande, ignorez simplement cet email.\n\n"
                        + "Cordialement,\n"
                        + "L'equipe Therapist AI",
                code
        );
    }

    public CodeDeliveryResult dispatchPasswordResetCode(String to, String code) {
        return send(
                to,
                "Therapist AI - Reinitialisation du mot de passe",
                "Bonjour,\n\n"
                        + "Vous avez demande la reinitialisation de votre mot de passe Therapist AI.\n\n"
                        + "Code de reinitialisation : " + code + "\n\n"
                        + "Ce code est valable pendant 10 minutes. Si vous n'avez pas fait cette demande, ignorez simplement cet email.\n\n"
                        + "Cordialement,\n"
                        + "L'equipe Therapist AI",
                code
        );
    }

    private void send(String to, String subject, String text) {
        send(to, subject, text, null);
    }

    private CodeDeliveryResult send(String to, String subject, String text, String code) {
        if (isBlank(to)) {
            throw new IllegalArgumentException("Adresse email destinataire manquante.");
        }
        if (!hasSmtpCredentials()) {
            return handleDeliveryFailure(
                    to,
                    subject,
                    text,
                    code,
                    null,
                    "Config SMTP incomplete. Renseigne MAIL_USERNAME et MAIL_PASSWORD, "
                            + "ou lance le backend avec scripts/run_backend.ps1."
            );
        }
        if (isBlank(smtpHost)) {
            return handleDeliveryFailure(
                    to,
                    subject,
                    text,
                    code,
                    null,
                    "Config SMTP incomplete. MAIL_HOST est manquant."
            );
        }
        try {
            SimpleMailMessage message = new SimpleMailMessage();
            message.setFrom(resolveFromAddress());
            message.setTo(to.trim());
            message.setSubject(subject);
            message.setText(text);
            mailSender.send(message);
            return CodeDeliveryResult.success();
        } catch (MailAuthenticationException ex) {
            return handleDeliveryFailure(
                    to,
                    subject,
                    text,
                    code,
                    ex,
                    authenticationFailureMessage()
            );
        } catch (Exception ex) {
            return handleDeliveryFailure(
                    to,
                    subject,
                    text,
                    code,
                    ex,
                    "Envoi email impossible. Verifie la config SMTP, le mot de passe d'application et MAIL_FROM."
            );
        }
    }

    private boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }

    private boolean hasSmtpCredentials() {
        return !isBlank(smtpUsername) && !isBlank(smtpPassword);
    }

    private String resolveFromAddress() {
        if (!isBlank(configuredFromAddress)) {
            return configuredFromAddress.trim();
        }
        if (!isBlank(smtpUsername)) {
            return smtpUsername.trim();
        }
        return "no-reply@therapist.local";
    }

    private String authenticationFailureMessage() {
        if (isGmailHost()) {
            return "Gmail a refuse l'authentification SMTP. Verifie MAIL_USERNAME, "
                    + "genere un nouveau mot de passe d'application Gmail et active la validation en deux etapes.";
        }
        return "Le serveur SMTP a refuse l'authentification. Verifie MAIL_USERNAME et MAIL_PASSWORD.";
    }

    private boolean isGmailHost() {
        return !isBlank(smtpHost) && smtpHost.toLowerCase().contains("gmail.com");
    }

    private CodeDeliveryResult handleDeliveryFailure(
            String to,
            String subject,
            String text,
            String code,
            Exception ex,
            String message
    ) {
        if (consoleFallback) {
            logFallbackMessage(to, subject, text, ex);
            return CodeDeliveryResult.fallback(
                    code,
                    message + " Le code a aussi ete affiche dans la console du backend."
            );
        }
        throw new IllegalStateException(message, ex);
    }

    private void logFallbackMessage(String to, String subject, String text, Exception ex) {
        System.out.println("[AUTH EMAIL FALLBACK] SMTP indisponible. Code affiche en console.");
        if (ex != null) {
            System.out.println("[AUTH EMAIL FALLBACK] Cause: " + ex.getMessage());
            if (isAuthenticationFailure(ex)) {
                System.out.println("[AUTH EMAIL FALLBACK] Conseil: regenere un mot de passe d'application Gmail et verifie la validation en deux etapes.");
            }
        }
        if (!hasSmtpCredentials()) {
            System.out.println("[AUTH EMAIL FALLBACK] Conseil: configure MAIL_USERNAME et MAIL_PASSWORD.");
        }
        System.out.println("[AUTH EMAIL FALLBACK] SMTP host: " + (isBlank(smtpHost) ? "<missing>" : smtpHost));
        System.out.println("[AUTH EMAIL FALLBACK] From: " + resolveFromAddress());
        System.out.println("[AUTH EMAIL FALLBACK] To: " + to);
        System.out.println("[AUTH EMAIL FALLBACK] Subject: " + subject);
        System.out.println("[AUTH EMAIL FALLBACK] Body: " + text);
    }

    private boolean isAuthenticationFailure(Throwable ex) {
        Throwable current = ex;
        while (current != null) {
            String message = current.getMessage();
            if (message != null) {
                String normalized = message.toLowerCase();
                if (normalized.contains("authentication failed")
                        || normalized.contains("username and password not accepted")
                        || normalized.contains("badcredentials")
                        || normalized.contains("5.7.8")) {
                    return true;
                }
            }
            current = current.getCause();
        }
        return false;
    }
}
