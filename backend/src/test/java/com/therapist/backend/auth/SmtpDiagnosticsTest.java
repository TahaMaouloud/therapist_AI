package com.therapist.backend.auth;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSenderImpl;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Properties;

@Disabled("Diagnostic SMTP local uniquement")
class SmtpDiagnosticsTest {
    @Test
    void canAuthenticateWithConfiguredSmtp() throws IOException {
        MailSettings settings = MailSettings.load();

        JavaMailSenderImpl sender = new JavaMailSenderImpl();
        sender.setHost(settings.host());
        sender.setPort(settings.port());
        sender.setUsername(settings.username());
        sender.setPassword(settings.password());

        Properties props = sender.getJavaMailProperties();
        props.setProperty("mail.smtp.auth", "true");
        props.setProperty("mail.smtp.starttls.enable", "true");
        props.setProperty("mail.smtp.starttls.required", "true");
        props.setProperty("mail.smtp.connectiontimeout", "5000");
        props.setProperty("mail.smtp.timeout", "10000");
        props.setProperty("mail.smtp.writetimeout", "10000");
        props.setProperty("mail.debug", "true");

        SimpleMailMessage message = new SimpleMailMessage();
        message.setFrom(settings.from());
        message.setTo(settings.username());
        message.setSubject("Therapist SMTP diagnostic");
        message.setText("SMTP diagnostic test.");

        sender.send(message);
    }

    private record MailSettings(String host, int port, String username, String password, String from) {
        private static MailSettings load() throws IOException {
            Path[] candidates = {
                    Path.of(".env"),
                    Path.of("..", ".env")
            };

            String host = "";
            String port = "";
            String username = "";
            String password = "";
            String from = "";

            for (Path candidate : candidates) {
                if (!Files.exists(candidate)) {
                    continue;
                }
                List<String> lines = Files.readAllLines(candidate);
                for (String rawLine : lines) {
                    String line = rawLine == null ? "" : rawLine.trim();
                    if (line.isEmpty() || line.startsWith("#") || !line.contains("=")) {
                        continue;
                    }
                    int separator = line.indexOf('=');
                    String key = line.substring(0, separator).trim();
                    String value = line.substring(separator + 1).trim();
                    switch (key) {
                        case "MAIL_HOST" -> host = value;
                        case "MAIL_PORT" -> port = value;
                        case "MAIL_USERNAME" -> username = value;
                        case "MAIL_PASSWORD" -> password = value;
                        case "MAIL_FROM" -> from = value;
                        default -> {
                        }
                    }
                }
            }

            return new MailSettings(
                    host,
                    port.isBlank() ? 587 : Integer.parseInt(port),
                    username,
                    password,
                    from.isBlank() ? username : from
            );
        }
    }
}
