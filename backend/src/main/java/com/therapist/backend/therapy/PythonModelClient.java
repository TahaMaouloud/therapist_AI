package com.therapist.backend.therapy;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.net.URI;
import java.net.ConnectException;
import java.net.http.HttpTimeoutException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.UUID;

@Service
public class PythonModelClient {
    private final HttpClient httpClient = HttpClient.newHttpClient();
    private final ObjectMapper objectMapper = new ObjectMapper();

    @Value("${python.service.base-url}")
    private String pythonBaseUrl;
    @Value("${python.service.text-timeout-seconds:420}")
    private long textTimeoutSeconds;
    @Value("${python.service.audio-timeout-seconds:300}")
    private long audioTimeoutSeconds;

    public JsonNode textSession(
            String text,
            String sessionId,
            List<TherapyDtos.ChatTurn> history,
            String authHeader
    ) throws IOException, InterruptedException {
        String body = objectMapper.writeValueAsString(new TextPayload(text, sessionId, history));
        HttpRequest.Builder builder = HttpRequest.newBuilder()
                .uri(URI.create(pythonBaseUrl + "/session/text"))
                .header("Content-Type", MediaType.APPLICATION_JSON_VALUE)
                .timeout(Duration.ofSeconds(textTimeoutSeconds))
                .POST(HttpRequest.BodyPublishers.ofString(body));
        if (authHeader != null && !authHeader.isBlank()) {
            builder.header("Authorization", authHeader);
        }
        HttpRequest request = builder.build();
        HttpResponse<String> response = sendRequest(
                request,
                "texte",
                textTimeoutSeconds,
                "python.service.text-timeout-seconds"
        );
        ensureSuccess(response, "texte");
        return parseBody(response.body());
    }

    public JsonNode audioUpload(
            MultipartFile audio,
            String sessionId,
            String authHeader
    ) throws IOException, InterruptedException {
        String boundary = "----TherapistBoundary" + UUID.randomUUID();
        byte[] body = buildMultipartBody(boundary, audio, sessionId);

        HttpRequest.Builder builder = HttpRequest.newBuilder()
                .uri(URI.create(pythonBaseUrl + "/session/audio-upload"))
                .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                .timeout(Duration.ofSeconds(audioTimeoutSeconds))
                .POST(HttpRequest.BodyPublishers.ofByteArray(body));
        if (authHeader != null && !authHeader.isBlank()) {
            builder.header("Authorization", authHeader);
        }

        HttpResponse<String> response = sendRequest(
                builder.build(),
                "upload audio",
                audioTimeoutSeconds,
                "python.service.audio-timeout-seconds"
        );
        ensureSuccess(response, "upload audio");
        return parseBody(response.body());
    }

    private HttpResponse<String> sendRequest(
            HttpRequest request,
            String operation,
            long timeoutSeconds,
            String timeoutSetting
    )
            throws IOException, InterruptedException {
        try {
            return httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        } catch (HttpTimeoutException ex) {
            throw new IOException(
                    "Echec appel service Python (" + operation + "): timeout apres "
                            + timeoutSeconds + "s. "
                            + "Reduis la charge du modele local ou augmente " + timeoutSetting + ".",
                    ex
            );
        } catch (ConnectException ex) {
            throw new IOException(
                    "Service Python IA indisponible sur " + pythonBaseUrl
                            + ". Demarre-le avec: uvicorn src.api.server:app --reload --port 8000",
                    ex
            );
        } catch (IOException ex) {
            String message = ex.getMessage();
            if (message == null || message.isBlank()) {
                message = ex.getClass().getSimpleName();
            }
            throw new IOException("Echec appel service Python (" + operation + "): " + message, ex);
        }
    }

    private void ensureSuccess(HttpResponse<String> response, String operation) throws IOException {
        int status = response.statusCode();
        if (status >= 200 && status < 300) {
            return;
        }
        String detail = extractDetail(response.body());
        if (detail.isBlank()) {
            throw new IOException("Service Python a retourne HTTP " + status + " (" + operation + ").");
        }
        throw new IOException("Service Python a retourne HTTP " + status + " (" + operation + "): " + detail);
    }

    private JsonNode parseBody(String body) throws IOException {
        if (body == null || body.isBlank()) {
            return objectMapper.createObjectNode();
        }
        return objectMapper.readTree(body);
    }

    private String extractDetail(String body) {
        if (body == null || body.isBlank()) {
            return "";
        }
        try {
            JsonNode json = objectMapper.readTree(body);
            JsonNode detail = json.path("detail");
            if (!detail.isMissingNode() && !detail.isNull()) {
                if (detail.isTextual()) {
                    return detail.asText("");
                }
                return detail.toString();
            }
        } catch (Exception ignored) {
            // fallback below: return raw response body
        }
        return body.strip();
    }

    private byte[] buildMultipartBody(String boundary, MultipartFile file, String sessionId) throws IOException {
        String filename = file.getOriginalFilename() == null ? "audio.wav" : file.getOriginalFilename();
        String partHeader = "--" + boundary + "\r\n"
                + "Content-Disposition: form-data; name=\"audio\"; filename=\"" + filename + "\"\r\n"
                + "Content-Type: " + (file.getContentType() == null ? "application/octet-stream" : file.getContentType()) + "\r\n\r\n";
        String sessionPart = "";
        if (sessionId != null && !sessionId.isBlank()) {
            sessionPart =
                    "\r\n--" + boundary + "\r\n"
                            + "Content-Disposition: form-data; name=\"session_id\"\r\n\r\n"
                            + sessionId.strip();
        }
        String ending = "\r\n--" + boundary + "--\r\n";

        byte[] headerBytes = partHeader.getBytes(StandardCharsets.UTF_8);
        byte[] fileBytes = file.getBytes();
        byte[] sessionBytes = sessionPart.getBytes(StandardCharsets.UTF_8);
        byte[] endingBytes = ending.getBytes(StandardCharsets.UTF_8);

        byte[] result = new byte[headerBytes.length + fileBytes.length + sessionBytes.length + endingBytes.length];
        System.arraycopy(headerBytes, 0, result, 0, headerBytes.length);
        System.arraycopy(fileBytes, 0, result, headerBytes.length, fileBytes.length);
        System.arraycopy(sessionBytes, 0, result, headerBytes.length + fileBytes.length, sessionBytes.length);
        System.arraycopy(
                endingBytes,
                0,
                result,
                headerBytes.length + fileBytes.length + sessionBytes.length,
                endingBytes.length
        );
        return result;
    }

    private record TextPayload(String text, String session_id, List<TherapyDtos.ChatTurn> history) {}
}
