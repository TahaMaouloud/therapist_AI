package com.therapist.backend.therapy;

import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface TherapyHistoryRepository extends JpaRepository<TherapyHistoryEntity, String> {
    List<TherapyHistoryEntity> findTop200ByUserIdOrderByPinnedDescCreatedAtDesc(String userId);
    Optional<TherapyHistoryEntity> findByIdAndUserId(String id, String userId);
    Optional<TherapyHistoryEntity> findByUserIdAndSessionId(String userId, String sessionId);
}
