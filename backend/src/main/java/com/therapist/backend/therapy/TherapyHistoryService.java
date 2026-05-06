package com.therapist.backend.therapy;

import org.springframework.stereotype.Service;

import java.time.Instant;
import java.util.List;
import java.util.UUID;

@Service
public class TherapyHistoryService {
    private final TherapyHistoryRepository therapyHistoryRepository;

    public TherapyHistoryService(TherapyHistoryRepository therapyHistoryRepository) {
        this.therapyHistoryRepository = therapyHistoryRepository;
    }

    public void saveTextSession(
            String userId,
            String sessionId,
            String userText,
            String emotion,
            String replyText
    ) {
        if (userId == null || userId.isBlank()) return;
        String normalizedSessionId = normalizeSessionId(sessionId);
        TherapyHistoryEntity entry = loadOrCreateBySession(userId, normalizedSessionId);

        entry.setSessionId(normalizedSessionId);
        entry.setChannel("text");
        entry.setUserText(userText);
        entry.setTranscript(null);
        entry.setEmotion(emotion);
        entry.setReplyText(replyText);
        entry.setCreatedAt(Instant.now());
        if (isBlank(entry.getChatTitle())) {
            entry.setChatTitle(buildDefaultTitle(userText));
        }
        therapyHistoryRepository.save(entry);
    }

    public void saveAudioSession(
            String userId,
            String sessionId,
            String transcript,
            String emotion,
            String replyText
    ) {
        if (userId == null || userId.isBlank()) return;
        String normalizedSessionId = normalizeSessionId(sessionId);
        TherapyHistoryEntity entry = loadOrCreateBySession(userId, normalizedSessionId);

        entry.setSessionId(normalizedSessionId);
        entry.setChannel("audio");
        entry.setUserText(null);
        entry.setTranscript(transcript);
        entry.setEmotion(emotion);
        entry.setReplyText(replyText);
        entry.setCreatedAt(Instant.now());
        if (isBlank(entry.getChatTitle())) {
            entry.setChatTitle(buildDefaultTitle(transcript));
        }
        therapyHistoryRepository.save(entry);
    }

    public List<TherapyHistoryEntity> findRecentByUserId(String userId) {
        return therapyHistoryRepository.findTop200ByUserIdOrderByPinnedDescCreatedAtDesc(userId);
    }

    public TherapyHistoryEntity renameById(String userId, String historyId, String title) {
        TherapyHistoryEntity entry = requireOwnedEntry(userId, historyId);
        entry.setChatTitle(cleanTitle(title));
        return therapyHistoryRepository.save(entry);
    }

    public TherapyHistoryEntity pinById(String userId, String historyId, boolean pinned) {
        TherapyHistoryEntity entry = requireOwnedEntry(userId, historyId);
        entry.setPinned(pinned);
        return therapyHistoryRepository.save(entry);
    }

    public void deleteById(String userId, String historyId) {
        TherapyHistoryEntity entry = requireOwnedEntry(userId, historyId);
        therapyHistoryRepository.delete(entry);
    }

    private TherapyHistoryEntity requireOwnedEntry(String userId, String historyId) {
        return therapyHistoryRepository.findByIdAndUserId(historyId, userId)
                .orElseThrow(() -> new IllegalArgumentException("Historique introuvable."));
    }

    private String buildDefaultTitle(String rawText) {
        String cleaned = cleanTitle(rawText);
        if (cleaned.isBlank()) return "Discussion";
        if (cleaned.length() <= 60) return cleaned;
        return cleaned.substring(0, 60).trim() + "...";
    }

    private String cleanTitle(String rawText) {
        if (rawText == null) return "";
        return rawText
                .replaceAll("\\s+", " ")
                .replaceAll("[\\r\\n\\t]", " ")
                .trim();
    }

    private TherapyHistoryEntity loadOrCreateBySession(String userId, String sessionId) {
        return therapyHistoryRepository.findByUserIdAndSessionId(userId, sessionId)
                .orElseGet(() -> {
                    TherapyHistoryEntity entry = new TherapyHistoryEntity();
                    entry.setId(UUID.randomUUID().toString());
                    entry.setUserId(userId);
                    entry.setPinned(false);
                    return entry;
                });
    }

    private String normalizeSessionId(String sessionId) {
        if (sessionId == null) return UUID.randomUUID().toString();
        String clean = sessionId.replaceAll("\\s+", " ").trim();
        if (clean.isBlank()) return UUID.randomUUID().toString();
        if (clean.length() <= 200) return clean;
        return clean.substring(0, 200);
    }

    private boolean isBlank(String value) {
        return value == null || value.isBlank();
    }
}
