package com.therapist.backend.therapy;

import jakarta.validation.constraints.NotBlank;
import java.util.List;

public class TherapyDtos {
    public record ChatTurn(String role, String content) {}
    public record TextRequest(
            @NotBlank String text,
            String session_id,
            List<ChatTurn> history
    ) {}
    public record RenameHistoryRequest(@NotBlank String title) {}
    public record PinHistoryRequest(boolean pinned) {}
}
