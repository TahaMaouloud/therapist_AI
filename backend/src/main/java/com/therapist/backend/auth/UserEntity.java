package com.therapist.backend.auth;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.PrePersist;
import jakarta.persistence.PreUpdate;
import jakarta.persistence.Table;

import java.time.Instant;

@Entity
@Table(name = "users")
public class UserEntity {
    @Id
    private String id;

    @Column(nullable = false, unique = true)
    private String email;

    @Column(nullable = false, unique = true)
    private String username;

    @Column(nullable = false)
    private String sex;

    @Column(nullable = false)
    private Integer age;

    @Column(name = "password_hash", nullable = false)
    private String passwordHash;

    @Column(name = "photo_path", columnDefinition = "TEXT")
    private String photoPath;

    @Column(nullable = false)
    private Boolean verified;

    @Column(name = "email_code_hash")
    private String emailCodeHash;

    @Column(name = "email_code_expires_at")
    private Instant emailCodeExpiresAt;

    @Column(name = "login_code_hash")
    private String loginCodeHash;

    @Column(name = "login_code_expires_at")
    private Instant loginCodeExpiresAt;

    @Column(name = "reset_code_hash")
    private String resetCodeHash;

    @Column(name = "reset_code_expires_at")
    private Instant resetCodeExpiresAt;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public String getSex() {
        return sex;
    }

    public void setSex(String sex) {
        this.sex = sex;
    }

    public Integer getAge() {
        return age;
    }

    public void setAge(Integer age) {
        this.age = age;
    }

    public String getPasswordHash() {
        return passwordHash;
    }

    public void setPasswordHash(String passwordHash) {
        this.passwordHash = passwordHash;
    }

    public String getPhotoPath() {
        return photoPath;
    }

    public void setPhotoPath(String photoPath) {
        this.photoPath = photoPath;
    }

    public Boolean getVerified() {
        return verified;
    }

    public boolean isVerified() {
        if (verified != null) {
            return verified;
        }
        return emailCodeHash == null || emailCodeHash.isBlank();
    }

    public void setVerified(Boolean verified) {
        this.verified = verified;
    }

    public String getEmailCodeHash() {
        return emailCodeHash;
    }

    public void setEmailCodeHash(String emailCodeHash) {
        this.emailCodeHash = emailCodeHash;
    }

    public Instant getEmailCodeExpiresAt() {
        return emailCodeExpiresAt;
    }

    public void setEmailCodeExpiresAt(Instant emailCodeExpiresAt) {
        this.emailCodeExpiresAt = emailCodeExpiresAt;
    }

    public String getLoginCodeHash() {
        return loginCodeHash;
    }

    public void setLoginCodeHash(String loginCodeHash) {
        this.loginCodeHash = loginCodeHash;
    }

    public Instant getLoginCodeExpiresAt() {
        return loginCodeExpiresAt;
    }

    public void setLoginCodeExpiresAt(Instant loginCodeExpiresAt) {
        this.loginCodeExpiresAt = loginCodeExpiresAt;
    }

    public String getResetCodeHash() {
        return resetCodeHash;
    }

    public void setResetCodeHash(String resetCodeHash) {
        this.resetCodeHash = resetCodeHash;
    }

    public Instant getResetCodeExpiresAt() {
        return resetCodeExpiresAt;
    }

    public void setResetCodeExpiresAt(Instant resetCodeExpiresAt) {
        this.resetCodeExpiresAt = resetCodeExpiresAt;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(Instant createdAt) {
        this.createdAt = createdAt;
    }

    @PrePersist
    @PreUpdate
    public void applyDefaults() {
        if (sex == null || sex.isBlank()) {
            sex = "unknown";
        }
        if (age == null) {
            age = 0;
        }
        if (verified == null) {
            verified = emailCodeHash == null || emailCodeHash.isBlank();
        }
        if (createdAt == null) {
            createdAt = Instant.now();
        }
    }
}
