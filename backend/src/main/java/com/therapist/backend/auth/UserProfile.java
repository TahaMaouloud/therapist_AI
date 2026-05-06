package com.therapist.backend.auth;

import java.time.Instant;

public class UserProfile {
    private final String id;
    private final String email;
    private final String username;
    private final String sex;
    private final int age;
    private final String passwordHash;
    private String photoPath;
    private boolean verified;
    private final Instant createdAt;

    public UserProfile(
            String id,
            String email,
            String username,
            String sex,
            int age,
            String passwordHash,
            String photoPath,
            boolean verified,
            Instant createdAt
    ) {
        this.id = id;
        this.email = email;
        this.username = username;
        this.sex = sex;
        this.age = age;
        this.passwordHash = passwordHash;
        this.photoPath = photoPath;
        this.verified = verified;
        this.createdAt = createdAt;
    }

    public String getId() { return id; }
    public String getEmail() { return email; }
    public String getUsername() { return username; }
    public String getSex() { return sex; }
    public int getAge() { return age; }
    public String getPasswordHash() { return passwordHash; }
    public String getPhotoPath() { return photoPath; }
    public void setPhotoPath(String photoPath) { this.photoPath = photoPath; }
    public boolean isVerified() { return verified; }
    public void setVerified(boolean verified) { this.verified = verified; }
    public Instant getCreatedAt() { return createdAt; }
}
