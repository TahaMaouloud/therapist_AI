package com.therapist.backend.controller;

import com.fasterxml.jackson.databind.JsonNode;
import com.therapist.backend.auth.AuthService;
import com.therapist.backend.therapy.PythonModelClient;
import com.therapist.backend.therapy.TherapyHistoryEntity;
import com.therapist.backend.therapy.TherapyHistoryService;
import com.therapist.backend.therapy.TherapyDtos;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/session")
public class TherapyController {
    private final PythonModelClient pythonModelClient;
    private final AuthService authService;
    private final TherapyHistoryService therapyHistoryService;

    public TherapyController(
            PythonModelClient pythonModelClient,
            AuthService authService,
            TherapyHistoryService therapyHistoryService
    ) {
        this.pythonModelClient = pythonModelClient;
        this.authService = authService;
        this.therapyHistoryService = therapyHistoryService;
    }

    @PostMapping("/text")
    public ResponseEntity<?> text(@Valid @RequestBody TherapyDtos.TextRequest req) {
        try {
            JsonNode result = pythonModelClient.textSession(
                    req.text(),
                    req.session_id(),
                    req.history(),
                    null
            );
            return ResponseEntity.ok(result);
        } catch (Exception ex) {
            return internalError(ex);
        }
    }

    @PostMapping("/text-auth")
    public ResponseEntity<?> textAuth(
            @Valid @RequestBody TherapyDtos.TextRequest req,
            @RequestHeader(value = "Authorization", required = false) String authHeader
    ) {
        try {
            JsonNode result = pythonModelClient.textSession(
                    req.text(),
                    req.session_id(),
                    req.history(),
                    authHeader
            );
            String userId = resolveUserId(authHeader);
            String emotion = result.path("emotion").asText("neutral");
            String reply = result.path("reply").asText("");
            therapyHistoryService.saveTextSession(userId, req.session_id(), req.text(), emotion, reply);
            return ResponseEntity.ok(result);
        } catch (Exception ex) {
            return internalError(ex);
        }
    }

    @PostMapping("/audio-upload")
    public ResponseEntity<?> audioUpload(
            @RequestPart("audio") MultipartFile audio,
            @RequestParam(value = "session_id", required = false) String sessionId,
            @RequestHeader(value = "Authorization", required = false) String authHeader
    ) {
        try {
            JsonNode result = pythonModelClient.audioUpload(audio, sessionId, authHeader);
            String userId = resolveUserId(authHeader);
            String transcript = result.path("transcript").asText("");
            String emotion = result.path("emotion").asText("neutral");
            String reply = result.path("reply").asText("");
            therapyHistoryService.saveAudioSession(userId, sessionId, transcript, emotion, reply);
            return ResponseEntity.ok(result);
        } catch (Exception ex) {
            return internalError(ex);
        }
    }

    @GetMapping("/history")
    public ResponseEntity<?> history(
            @RequestHeader(value = "Authorization", required = false) String authHeader
    ) {
        String userId = resolveRequiredUserId(authHeader);
        if (userId == null) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("detail", "Session invalide."));
        }
        List<Map<String, Object>> items = therapyHistoryService.findRecentByUserId(userId).stream()
                .map(this::toMap)
                .collect(Collectors.toList());
        return ResponseEntity.ok(Map.of("items", items));
    }

    @PatchMapping("/history/{id}/rename")
    public ResponseEntity<?> renameHistory(
            @PathVariable String id,
            @Valid @RequestBody TherapyDtos.RenameHistoryRequest req,
            @RequestHeader(value = "Authorization", required = false) String authHeader
    ) {
        String userId = resolveRequiredUserId(authHeader);
        if (userId == null) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("detail", "Session invalide."));
        }
        try {
            TherapyHistoryEntity updated = therapyHistoryService.renameById(userId, id, req.title());
            return ResponseEntity.ok(Map.of("item", toMap(updated)));
        } catch (IllegalArgumentException ex) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(Map.of("detail", ex.getMessage()));
        }
    }

    @PatchMapping("/history/{id}/pin")
    public ResponseEntity<?> pinHistory(
            @PathVariable String id,
            @RequestBody TherapyDtos.PinHistoryRequest req,
            @RequestHeader(value = "Authorization", required = false) String authHeader
    ) {
        String userId = resolveRequiredUserId(authHeader);
        if (userId == null) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("detail", "Session invalide."));
        }
        try {
            TherapyHistoryEntity updated = therapyHistoryService.pinById(userId, id, req.pinned());
            return ResponseEntity.ok(Map.of("item", toMap(updated)));
        } catch (IllegalArgumentException ex) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(Map.of("detail", ex.getMessage()));
        }
    }

    @DeleteMapping("/history/{id}")
    public ResponseEntity<?> deleteHistory(
            @PathVariable String id,
            @RequestHeader(value = "Authorization", required = false) String authHeader
    ) {
        String userId = resolveRequiredUserId(authHeader);
        if (userId == null) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(Map.of("detail", "Session invalide."));
        }
        try {
            therapyHistoryService.deleteById(userId, id);
            return ResponseEntity.ok(Map.of("message", "Historique supprime."));
        } catch (IllegalArgumentException ex) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(Map.of("detail", ex.getMessage()));
        }
    }

    private Map<String, Object> toMap(TherapyHistoryEntity entry) {
        return Map.of(
                "id", entry.getId(),
                "session_id", entry.getSessionId() == null ? "" : entry.getSessionId(),
                "channel", entry.getChannel(),
                "title", entry.getChatTitle() == null ? "" : entry.getChatTitle(),
                "pinned", entry.isPinned(),
                "user_text", entry.getUserText() == null ? "" : entry.getUserText(),
                "transcript", entry.getTranscript() == null ? "" : entry.getTranscript(),
                "emotion", entry.getEmotion() == null ? "neutral" : entry.getEmotion(),
                "reply_text", entry.getReplyText() == null ? "" : entry.getReplyText(),
                "created_at", entry.getCreatedAt() == null ? "" : entry.getCreatedAt().toString()
        );
    }

    private String resolveRequiredUserId(String authHeader) {
        String token = extractBearer(authHeader);
        if (token == null) return null;
        return authService.resolveUserId(token);
    }

    private String resolveUserId(String authHeader) {
        String token = extractBearer(authHeader);
        if (token == null) return null;
        return authService.resolveUserId(token);
    }

    private String extractBearer(String authHeader) {
        if (authHeader == null || !authHeader.toLowerCase().startsWith("bearer ")) {
            return null;
        }
        return authHeader.substring(7).trim();
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
        return "Erreur interne du backend.";
    }
}
