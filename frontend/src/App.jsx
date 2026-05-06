import { useEffect, useMemo, useRef, useState } from "react";

const API_BASES = (() => {
  const envBase = String(import.meta.env.VITE_API_BASE_URL || "").trim();
  if (envBase) {
    return [envBase, "http://127.0.0.1:8080", "http://localhost:8080"];
  }
  return ["/api", "http://127.0.0.1:8080", "http://localhost:8080"];
})();
const TTS_PLAYBACK_MODE = String(import.meta.env.VITE_TTS_PLAYBACK || "browser")
  .trim()
  .toLowerCase();
const AUTH_TOKEN_STORAGE_KEY = "therapist_access_token";
const REMEMBERED_EMAILS_STORAGE_KEY = "therapist_remembered_emails";
const MAX_VOICE_SECONDS = 20;
const MODEL_HISTORY_MAX_ITEMS = 10;
const SESSION_WELCOME_VARIANTS = [
  {
    title: "Bienvenu dans ton espace de serenite",
    subtitle: "Respire doucement... Tu peux commencer quand tu veux, je suis la pour toi."
  },
  {
    title: "Content de te revoir, {name}.",
    subtitle: "Dis moi ce que tu ressens aujourd hui, on avance ensemble."
  },
  {
    title: "Bienvenue {name}, ton espace est pret.",
    subtitle: "Une petite phrase suffit pour commencer doucement."
  },
  {
    title: "Ravi de te retrouver, {name}.",
    subtitle: "On peut transformer une journee lourde en petits pas plus legers."
  },
  {
    title: "Salut {name}, je suis la pour toi.",
    subtitle: "Ecris ce que tu as sur le coeur, meme en quelques mots."
  }
];
const EMOTION_LABELS = {
  angry: "colere",
  calm: "calme",
  disgust: "degout",
  fear: "peur",
  fearful: "peur",
  happy: "joie",
  neutral: "neutre",
  sad: "tristesse",
  sadness: "tristesse",
  surprised: "surprise"
};

function loadStoredToken() {
  if (typeof window === "undefined") return "";
  try {
    return String(window.localStorage.getItem(AUTH_TOKEN_STORAGE_KEY) || "");
  } catch {
    return "";
  }
}

function saveStoredToken(token) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, String(token || ""));
  } catch {
    // no-op
  }
}

function clearStoredToken() {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
  } catch {
    // no-op
  }
}

function normalizeEmailForMemory(value) {
  return String(value || "").trim().toLowerCase();
}

function loadRememberedEmails() {
  if (typeof window === "undefined") return [];
  try {
    const raw = String(window.localStorage.getItem(REMEMBERED_EMAILS_STORAGE_KEY) || "");
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .map((entry) => normalizeEmailForMemory(entry))
      .filter(Boolean);
  } catch {
    return [];
  }
}

function saveRememberedEmail(email) {
  const normalized = normalizeEmailForMemory(email);
  if (!normalized || typeof window === "undefined") return;
  try {
    const current = new Set(loadRememberedEmails());
    current.add(normalized);
    window.localStorage.setItem(
      REMEMBERED_EMAILS_STORAGE_KEY,
      JSON.stringify(Array.from(current))
    );
  } catch {
    // no-op
  }
}

function removeRememberedEmail(email) {
  const normalized = normalizeEmailForMemory(email);
  if (!normalized || typeof window === "undefined") return;
  try {
    const next = loadRememberedEmails().filter((entry) => entry !== normalized);
    window.localStorage.setItem(
      REMEMBERED_EMAILS_STORAGE_KEY,
      JSON.stringify(next)
    );
  } catch {
    // no-op
  }
}

function pickSessionWelcome(userName) {
  const rawName = String(userName || "ami").trim();
  const firstName = rawName.split(/\s+/)[0] || "ami";
  const idx = Math.floor(Math.random() * SESSION_WELCOME_VARIANTS.length);
  const item = SESSION_WELCOME_VARIANTS[idx];
  return {
    title: item.title.replace("{name}", firstName),
    subtitle: item.subtitle
  };
}

function historyPreview(item) {
  const source = String(item?.user_text || item?.transcript || item?.reply_text || "");
  if (!source.trim()) return "Conversation with no text";
  return source.length > 56 ? `${source.slice(0, 56)}...` : source;
}

function historyTitle(item) {
  const serverTitle = String(item?.title || "").trim();
  if (serverTitle) return serverTitle;
  return historyPreview(item);
}

function formatHistoryDate(value) {
  if (!value) return "";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString("en-US", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });
}

function sortHistoryItems(items) {
  return [...items].sort((a, b) => {
    const pinA = Boolean(a?.pinned);
    const pinB = Boolean(b?.pinned);
    if (pinA !== pinB) return pinA ? -1 : 1;
    const tA = new Date(a?.created_at || 0).getTime();
    const tB = new Date(b?.created_at || 0).getTime();
    return tB - tA;
  });
}

function historyItemToMessages(item) {
  const emotion = String(item?.emotion || "neutral");
  const userText = String(item?.user_text || item?.transcript || "").trim();
  const reply = String(item?.reply_text || "").trim();
  const next = [];
  if (userText) {
    next.push({ role: "user", content: userText });
  }
  if (reply || !userText) {
    next.push({
      role: "assistant",
      content: reply || "I am here with you.",
      emotion
    });
  }
  return next;
}

function normalizeEmotionKey(emotion) {
  const key = String(emotion || "neutral").trim().toLowerCase();
  return key.replace(/[^a-z0-9_-]/g, "") || "neutral";
}

function formatEmotionLabel(emotion) {
  const key = normalizeEmotionKey(emotion);
  return EMOTION_LABELS[key] || key;
}

function formatMessageRole(role, userName) {
  const normalizedRole = String(role || "").trim().toLowerCase();
  if (normalizedRole === "assistant") {
    return "Therapist";
  }
  if (normalizedRole === "user") {
    const normalizedName = String(userName || "").trim();
    return normalizedName || "User";
  }
  if (normalizedRole === "system") {
    return "System";
  }
  return normalizedRole || "Message";
}

async function apiFetch(path, options = {}) {
  let lastError = null;
  let lastResponse = null;

  for (const base of API_BASES) {
    try {
      const cleanBase = String(base || "").replace(/\/+$/, "");
      const res = await fetch(`${cleanBase}${path}`, options);
      lastResponse = res;

      if (res.ok) {
        return res;
      }

      // Allow local dev fallback: /api may be proxied, but if path is not found there
      // try direct backend hosts. Keep non-404 responses so user sees actual errors.
      // In local dev, /api is proxied by Vite and should be rewritten to the
      // Spring backend root. If the proxy responds with a media-type mismatch,
      // try the direct backend hosts before surfacing the error.
      const isLocalProxyBase = cleanBase === "/api";
      if (res.status === 404 || (isLocalProxyBase && res.status === 415)) {
        continue;
      }

      return res;
    } catch (err) {
      lastError = err;
    }
  }

  if (lastResponse) {
    return lastResponse;
  }

  throw lastError || new Error("Failed to fetch");
}

function createClientSessionId() {
  const randomPart =
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `${Math.random().toString(36).slice(2)}${Date.now().toString(36)}`;
  return `chat-${randomPart}`;
}

function buildModelHistory(messages) {
  if (!Array.isArray(messages)) return [];
  return messages
    .filter((msg) => msg && (msg.role === "user" || msg.role === "assistant"))
    .map((msg) => ({
      role: msg.role,
      content: String(msg.content || "").trim()
    }))
    .filter((msg) => msg.content)
    .slice(-MODEL_HISTORY_MAX_ITEMS);
}

function App() {
  const [tab, setTab] = useState("login");
  const [token, setToken] = useState(() => loadStoredToken());
  const [me, setMe] = useState(null);
  const [status, setStatus] = useState("");
  const [isAuthBootstrapping, setIsAuthBootstrapping] = useState(() => Boolean(loadStoredToken()));
  const [verifyEmailValue, setVerifyEmailValue] = useState("");
  const [verifyCode, setVerifyCode] = useState("");
  const [isResendingVerification, setIsResendingVerification] = useState(false);
  const [loginEmail, setLoginEmail] = useState("");
  const [loginEmailPending, setLoginEmailPending] = useState("");
  const [loginCode, setLoginCode] = useState("");
  const [loginRememberMe, setLoginRememberMe] = useState(false);
  const [rememberedEmails, setRememberedEmails] = useState(() => loadRememberedEmails());
  const [loginHumanCheck, setLoginHumanCheck] = useState(false);
  const [loginPasswordVisible, setLoginPasswordVisible] = useState(false);
  const [isLoginSubmitting, setIsLoginSubmitting] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [chatWelcome, setChatWelcome] = useState(() => pickSessionWelcome("ami"));
  const [hasComposerInteracted, setHasComposerInteracted] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [voiceOpen, setVoiceOpen] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isVoiceSending, setIsVoiceSending] = useState(false);
  const [voiceSeconds, setVoiceSeconds] = useState(0);
  const [speakingIndex, setSpeakingIndex] = useState(null);
  const [screen, setScreen] = useState("dashboard");
  const [historyItems, setHistoryItems] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [activeHistoryId, setActiveHistoryId] = useState("");
  const [openHistoryMenuId, setOpenHistoryMenuId] = useState("");
  const [isUploadingPhoto, setIsUploadingPhoto] = useState(false);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [isCameraStarting, setIsCameraStarting] = useState(false);
  const [forgotPasswordEmail, setForgotPasswordEmail] = useState("");
  const [resetPasswordCode, setResetPasswordCode] = useState("");
  const [resetPasswordNew, setResetPasswordNew] = useState("");
  const [resetPasswordConfirm, setResetPasswordConfirm] = useState("");
  const [isForgotSubmitting, setIsForgotSubmitting] = useState(false);
  const [isResetSubmitting, setIsResetSubmitting] = useState(false);
  const [profileUsername, setProfileUsername] = useState("");
  const [profileOldPassword, setProfileOldPassword] = useState("");
  const [profileNewPassword, setProfileNewPassword] = useState("");
  const [profileOldPasswordVisible, setProfileOldPasswordVisible] = useState(false);
  const [profileNewPasswordVisible, setProfileNewPasswordVisible] = useState(false);
  const [isProfileSaving, setIsProfileSaving] = useState(false);
  const [isPasswordSaving, setIsPasswordSaving] = useState(false);
  const normalizedLoginEmail = normalizeEmailForMemory(loginEmail);
  const isLoginEmailRemembered =
    Boolean(normalizedLoginEmail) && rememberedEmails.includes(normalizedLoginEmail);

  const isAuthBackVisible = tab === "forgot-password" || tab === "reset-password";
  const switchAuthTab = (nextTab) => {
    setTab(nextTab);
    setStatus("");
  };
  const backToLogin = () => {
    switchAuthTab("login");
  };
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const voiceChunksRef = useRef([]);
  const voiceTimerRef = useRef(null);
  const serverAudioRef = useRef(null);
  const profileFileInputRef = useRef(null);
  const profileCameraVideoRef = useRef(null);
  const profileCameraStreamRef = useRef(null);
  const llmSessionIdRef = useRef(createClientSessionId());

  const headers = useMemo(() => {
    const base = { Accept: "application/json" };
    if (token) base.Authorization = `Bearer ${token}`;
    return base;
  }, [token]);

  function resetSessionWelcome(userName) {
    setChatWelcome(pickSessionWelcome(userName || me?.username || "ami"));
    setHasComposerInteracted(false);
  }

  function addMessage(role, content, extra = {}) {
    setMessages((prev) => [...prev, { role, content, ...extra }]);
  }

  function addAssistantMessage(content, extra = {}, autoSpeak = false) {
    const message = { role: "assistant", content, ...extra };
    setMessages((prev) => [...prev, message]);
    if (autoSpeak) {
      // Delay one tick so the message exists before triggering playback.
      setTimeout(() => {
        speakMessage(message, -1);
      }, 0);
    }
  }

  async function readApiPayload(res) {
    const raw = await res.text();
    if (!raw) return {};
    try {
      return JSON.parse(raw);
    } catch {
      return { detail: raw };
    }
  }

  function extractApiError(data, fallback, status) {
    const detail = data?.detail || data?.error || data?.message;
    if (detail) {
      const text = String(detail).trim();
      const lower = text.toLowerCase();
      if (
        lower.startsWith("<!doctype") ||
        lower.startsWith("<html") ||
        lower.includes("<body") ||
        lower.includes("<head")
      ) {
        return status
          ? `${fallback} (HTTP ${status}). Check that the Spring backend is running on 127.0.0.1:8080.`
          : `${fallback}. Check that the Spring backend is running on 127.0.0.1:8080.`;
      }
      return text;
    }
    if (status) return `${fallback} (HTTP ${status})`;
    return fallback;
  }

  function normalizeClientError(err) {
    const message = String(err?.message || err || "");
    const lower = message.toLowerCase();
    if (message.includes("Failed to execute 'json' on 'Response'")) {
      return "Le backend a renvoye une reponse invalide. Verifie que Spring tourne sur 127.0.0.1:8080.";
    }
    if (message === "Failed to fetch") {
      return "Connexion au backend impossible. Verifie que 127.0.0.1:8080 ou localhost:8080 tourne.";
    }
    if (
      lower.includes("envoi email impossible") ||
      lower.includes("gmail a refuse l'authentification smtp") ||
      lower.includes("authentification smtp") ||
      lower.includes("mot de passe d'application")
    ) {
      return "Le SMTP a echoue. Le code n'a pas ete envoye par email. Verifie la configuration SMTP, Gmail et le mot de passe d'application.";
    }
    return message;
  }

  function buildDeliveryStatus(data, fallbackMessage) {
    const message = String(data?.message || fallbackMessage || "").trim();
    const detail = String(data?.delivery_detail || "").trim();
    if (message) return message;
    if (detail) return detail;
    return fallbackMessage || "";
  }

  function stopSpeech() {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    if (serverAudioRef.current) {
      try {
        serverAudioRef.current.pause();
        serverAudioRef.current.currentTime = 0;
      } catch {
        // no-op
      }
      serverAudioRef.current = null;
    }
    setSpeakingIndex(null);
  }

  function stopProfileCameraStream() {
    if (profileCameraStreamRef.current) {
      profileCameraStreamRef.current.getTracks().forEach((track) => track.stop());
      profileCameraStreamRef.current = null;
    }
    if (profileCameraVideoRef.current) {
      profileCameraVideoRef.current.srcObject = null;
    }
  }

  function speakMessage(message, index) {
    if (speakingIndex === index) {
      stopSpeech();
      return;
    }

    stopSpeech();

    const messageText = String(message?.content || "").trim() || "I am here with you.";
    const ttsAudioBase64 = message?.ttsAudioBase64 || "";
    const ttsAudioMime = message?.ttsAudioMime || "audio/wav";
    const canUseBrowserTts = "speechSynthesis" in window;
    const playBrowserTts = () => {
      if (!canUseBrowserTts) return false;
      const utterance = new SpeechSynthesisUtterance(messageText);
      utterance.lang = "en-US";
      utterance.rate = 0.98;
      utterance.onend = () => setSpeakingIndex(null);
      utterance.onerror = () => setSpeakingIndex(null);
      setSpeakingIndex(index);
      window.speechSynthesis.speak(utterance);
      return true;
    };

    // Browser speech is usually more natural than server WAV in Docker/Linux.
    if (TTS_PLAYBACK_MODE !== "server" && playBrowserTts()) {
      return;
    }

    if (ttsAudioBase64) {
      const audio = new Audio(`data:${ttsAudioMime};base64,${ttsAudioBase64}`);
      serverAudioRef.current = audio;
      audio.onended = () => {
        if (serverAudioRef.current === audio) {
          serverAudioRef.current = null;
        }
        setSpeakingIndex(null);
      };
      audio.onerror = () => {
        if (serverAudioRef.current === audio) {
          serverAudioRef.current = null;
        }
        setSpeakingIndex(null);
        if (!playBrowserTts()) {
          setStatus("La lecture audio du serveur a echoue.");
        }
      };
      setSpeakingIndex(index);
      audio.play().catch(() => {
        setSpeakingIndex(null);
        if (!playBrowserTts()) {
          setStatus("La lecture audio a ete bloquee par le navigateur.");
        }
      });
      return;
    }

    if (!playBrowserTts()) {
      setStatus("La synthese vocale du navigateur n'est pas prise en charge.");
      return;
    }
  }

  useEffect(() => {
    return () => {
      if (voiceTimerRef.current) {
        clearInterval(voiceTimerRef.current);
      }
      stopSpeech();
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      stopProfileCameraStream();
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function restoreSession() {
      if (!token) {
        setIsAuthBootstrapping(false);
        return;
      }
      if (me) {
        setIsAuthBootstrapping(false);
        return;
      }

      setIsAuthBootstrapping(true);
      try {
        const res = await apiFetch("/auth/me", {
          headers: {
            Accept: "application/json",
            Authorization: `Bearer ${token}`
          }
        });
        const data = await readApiPayload(res);
        if (!res.ok) {
          throw new Error(extractApiError(data, "Erreur de session", res.status));
        }
        if (cancelled) return;

        const user = data?.user || null;
        if (!user) {
          throw new Error("Session invalide.");
        }
        setMe(user);
        setScreen("chat");
        setTab("therapy");
        resetSessionWelcome(user?.username);
        setStatus("");
      } catch {
        if (cancelled) return;
        clearStoredToken();
        setToken("");
        setMe(null);
        setStatus("");
        setTab("login");
      } finally {
        if (!cancelled) {
          setIsAuthBootstrapping(false);
        }
      }
    }

    restoreSession();
    return () => {
      cancelled = true;
    };
  }, [token, me]);

  useEffect(() => {
    if (!token) {
      setHistoryItems([]);
      setHistoryLoading(false);
      setActiveHistoryId("");
      setOpenHistoryMenuId("");
      setIsCameraOpen(false);
      stopProfileCameraStream();
      return;
    }
    loadHistory(true);
  }, [token]);

  useEffect(() => {
    setProfileUsername(String(me?.username || ""));
  }, [me?.username]);

  useEffect(() => {
    if (!isCameraOpen || !profileCameraVideoRef.current || !profileCameraStreamRef.current) {
      return undefined;
    }
    const video = profileCameraVideoRef.current;
    video.srcObject = profileCameraStreamRef.current;
    video.play().catch(() => {
      setStatus("Unable to start camera preview.");
    });
    return undefined;
  }, [isCameraOpen]);

  function formatDuration(seconds) {
    const m = String(Math.floor(seconds / 60)).padStart(2, "0");
    const s = String(seconds % 60).padStart(2, "0");
    return `${m}:${s}`;
  }

  function openVerificationFlow(email, message) {
    setVerifyEmailValue(String(email || "").trim());
    setVerifyCode("");
    setTab("verify");
    setStatus(message);
  }

  async function uploadProfilePhoto(file) {
    if (!file || !token) return;
    if (!file.type.startsWith("image/")) {
      setStatus("Choose an image file (JPG, PNG, WEBP, GIF).");
      return;
    }
    if (file.size > 2_500_000) {
      setStatus("Image is too large. Maximum 2.5 MB.");
      return;
    }
    setIsUploadingPhoto(true);
    try {
      const payload = new FormData();
      payload.append("photo", file);
      const res = await apiFetch("/auth/profile-photo", {
        method: "POST",
        headers,
        body: payload
      });
      const data = await readApiPayload(res);
      if (!res.ok) {
        throw new Error(extractApiError(data, "Profile photo upload error", res.status));
      }
      setMe(data.user);
      setStatus("Profile photo updated.");
    } catch (err) {
      setStatus(normalizeClientError(err));
    } finally {
      setIsUploadingPhoto(false);
    }
  }

  async function handleProfilePhotoChange(e) {
    const file = e.target.files?.[0];
    if (!file || !token) return;
    try {
      await uploadProfilePhoto(file);
    } finally {
      e.target.value = "";
    }
  }

  async function openProfileCamera() {
    if (isUploadingPhoto || isCameraStarting) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("Camera is not supported on this browser.");
      return;
    }
    setIsCameraStarting(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false
      });
      stopProfileCameraStream();
      profileCameraStreamRef.current = stream;
      setIsCameraOpen(true);
    } catch (err) {
      setStatus(normalizeClientError(err));
      stopProfileCameraStream();
    } finally {
      setIsCameraStarting(false);
    }
  }

  function closeProfileCamera() {
    setIsCameraOpen(false);
    stopProfileCameraStream();
  }

  async function captureProfilePhotoFromCamera() {
    const video = profileCameraVideoRef.current;
    if (!video) return;

    const width = video.videoWidth || 0;
    const height = video.videoHeight || 0;
    if (!width || !height) {
      setStatus("Camera is not ready yet. Try again in 1 second.");
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      setStatus("Camera capture unavailable.");
      return;
    }
    ctx.drawImage(video, 0, 0, width, height);

    const blob = await new Promise((resolve) => {
      canvas.toBlob((result) => resolve(result), "image/jpeg", 0.92);
    });
    if (!blob) {
      setStatus("Photo capture failed.");
      return;
    }

    const file = new File([blob], `profile-${Date.now()}.jpg`, { type: "image/jpeg" });
    await uploadProfilePhoto(file);
    closeProfileCamera();
  }

  function triggerLocalPhotoImport() {
    if (isUploadingPhoto) return;
    profileFileInputRef.current?.click();
  }

  async function deleteProfilePhoto() {
    if (!token) return;
    if (!me?.photo_path) {
      setStatus("No profile photo to delete.");
      return;
    }

    setIsUploadingPhoto(true);
    try {
      const res = await apiFetch("/auth/profile-photo", {
        method: "DELETE",
        headers
      });
      const data = await readApiPayload(res);
      if (!res.ok) {
        throw new Error(extractApiError(data, "Profile photo delete error", res.status));
      }
      setMe(data.user);
      setStatus("Profile photo deleted.");
    } catch (err) {
      setStatus(normalizeClientError(err));
    } finally {
      setIsUploadingPhoto(false);
    }
  }

  async function deleteAccount() {
    if (!token) return;
    if (!window.confirm("Voulez-vous vraiment supprimer votre compte ? Cette action est irreversible.")) {
      return;
    }

    setStatus("Suppression du compte en cours...");
    try {
      const res = await apiFetch("/auth/account", {
        method: "DELETE",
        headers
      });
      const data = await readApiPayload(res);
      if (!res.ok) {
        throw new Error(extractApiError(data, "Account delete error", res.status));
      }
      logout();
      setStatus(data?.message || "Compte supprime.");
    } catch (err) {
      setStatus(normalizeClientError(err));
    }
  }

  async function saveProfileUsername(e) {
    e.preventDefault();
    if (!token || isProfileSaving) return;
    const nextUsername = String(profileUsername || "").trim();
    if (!nextUsername) {
      setStatus("Le nom d'utilisateur est obligatoire.");
      return;
    }

    setIsProfileSaving(true);
    setStatus("");
    try {
      const res = await apiFetch("/auth/profile", {
        method: "PATCH",
        headers: { ...headers, "Content-Type": "application/json" },
        body: JSON.stringify({ username: nextUsername })
      });
      const data = await readApiPayload(res);
      if (!res.ok) {
        throw new Error(extractApiError(data, "Erreur de mise a jour du profil", res.status));
      }
      setMe(data?.user || me);
      setStatus(data?.message || "Profil mis a jour.");
    } catch (err) {
      setStatus(normalizeClientError(err));
    } finally {
      setIsProfileSaving(false);
    }
  }

  async function saveProfilePassword(e) {
    e.preventDefault();
    if (!token || isPasswordSaving) return;
    const oldPassword = String(profileOldPassword || "");
    const newPassword = String(profileNewPassword || "");
    if (!oldPassword || !newPassword) {
      setStatus("L'ancien et le nouveau mot de passe sont obligatoires.");
      return;
    }

    setIsPasswordSaving(true);
    setStatus("");
    try {
      const res = await apiFetch("/auth/change-password", {
        method: "POST",
        headers: { ...headers, "Content-Type": "application/json" },
        body: JSON.stringify({
          oldPassword,
          newPassword
        })
      });
      const data = await readApiPayload(res);
      if (!res.ok) {
        throw new Error(extractApiError(data, "Erreur de mise a jour du mot de passe", res.status));
      }
      setProfileOldPassword("");
      setProfileNewPassword("");
      setProfileOldPasswordVisible(false);
      setProfileNewPasswordVisible(false);
      setStatus(data?.message || "Mot de passe mis a jour.");
    } catch (err) {
      setStatus(normalizeClientError(err));
    } finally {
      setIsPasswordSaving(false);
    }
  }

  async function registerUser(e) {
    e.preventDefault();
    setStatus("");
    const fd = new FormData(e.currentTarget);
    const email = String(fd.get("email") || "").trim();
    const confirmEmail = String(fd.get("confirmEmail") || "").trim();

    if (email !== confirmEmail) {
      setStatus("L'email et sa confirmation ne correspondent pas.");
      return;
    }

    try {
      const payload = new FormData();
      payload.append("email", email);
      payload.append("username", String(fd.get("username") || ""));
      payload.append("sex", String(fd.get("sex") || ""));
      payload.append("age", String(fd.get("age") || ""));
      payload.append("password", String(fd.get("password") || ""));
      payload.append("confirm_password", String(fd.get("confirmPassword") || ""));

      const res = await apiFetch("/auth/register", {
        method: "POST",
        body: payload
      });
      const data = await readApiPayload(res);
      if (!res.ok) {
        throw new Error(extractApiError(data, "Erreur d'inscription", res.status));
      }

      openVerificationFlow(
        email,
        buildDeliveryStatus(
          data,
          "Le code a ete envoye au client par email. Saisis-le ici pour verifier le compte."
        )
      );
    } catch (err) {
      const message = normalizeClientError(err);
      if (String(message).toLowerCase().includes("email deja utilise")) {
        openVerificationFlow(
          email,
          "Cet email existe deja. Si le compte n'est pas encore verifie, le client doit entrer le code recu par email ou cliquer sur 'Resend code'."
        );
      } else {
        setStatus(message);
      }
    }
  }

  async function verifyEmail(e) {
    e.preventDefault();
    setStatus("");

    try {
      const res = await apiFetch("/auth/verify-email", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: verifyEmailValue.trim(),
          code: verifyCode.trim()
        })
      });
      const data = await readApiPayload(res);
      if (!res.ok) {
        throw new Error(extractApiError(data, "Erreur de verification", res.status));
      }

      setStatus("Email verifie. Vous pouvez maintenant vous connecter.");
      setTab("login");
    } catch (err) {
      setStatus(normalizeClientError(err));
    }
  }

  async function resendVerificationCode() {
    const email = verifyEmailValue.trim();
    if (!email || isResendingVerification) return;

    setStatus("Envoi d'un nouveau code de verification...");
    setIsResendingVerification(true);
    try {
      const res = await apiFetch("/auth/resend-verification", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email })
      });
      const data = await readApiPayload(res);
      if (!res.ok) {
        throw new Error(extractApiError(data, "Erreur de renvoi du code", res.status));
      }
      setStatus(buildDeliveryStatus(data, "Un nouveau code de verification a ete envoye."));
    } catch (err) {
      setStatus(normalizeClientError(err));
    } finally {
      setIsResendingVerification(false);
    }
  }

  async function login(e) {
    e.preventDefault();
    if (isLoginSubmitting) return;
    setStatus("");
    setIsLoginSubmitting(true);
    setStatus("Connexion en cours, merci de patienter...");
    const fd = new FormData(e.currentTarget);
    const email = normalizeEmailForMemory(fd.get("email"));
    const password = String(fd.get("password") || "");
    const rememberMeChecked = Boolean(fd.get("rememberMe"));
    const rememberMe = isLoginEmailRemembered || rememberMeChecked;
    const humanCheck = Boolean(fd.get("humanCheck"));
    setLoginRememberMe(rememberMe);
    setLoginHumanCheck(humanCheck);

    if (!humanCheck) {
      setStatus("Veuillez cocher \"Je ne suis pas un robot\".");
      setIsLoginSubmitting(false);
      return;
    }

    try {
      const res = await apiFetch("/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email,
          password,
          remember_me: rememberMe,
          human_check: humanCheck
        })
      });
      const data = await readApiPayload(res);
      if (!res.ok) {
        throw new Error(extractApiError(data, "Erreur de connexion", res.status));
      }

      if (Boolean(data.requires_2fa)) {
        setLoginEmailPending(email);
        setLoginCode("");
        setStatus(buildDeliveryStatus(data, "Code de connexion envoye par email."));
        setTab("login-code");
        return;
      }

      const accessToken = String(data.access_token || "").trim();
      if (!accessToken) {
        throw new Error("La reponse de connexion ne contient pas de jeton d'acces.");
      }
      if (rememberMe) {
        saveStoredToken(accessToken);
        saveRememberedEmail(email);
        setRememberedEmails(loadRememberedEmails());
      } else {
        clearStoredToken();
      }
      setToken(accessToken);
      setMe(data.user || null);
      setStatus(data.message || "Connexion reussie.");
      setScreen("chat");
      setTab("therapy");
      setMessages([]);
      setChatInput("");
      llmSessionIdRef.current = createClientSessionId();
      resetSessionWelcome(data?.user?.username);
      setActiveHistoryId("");
      setLoginEmailPending("");
      setLoginCode("");
    } catch (err) {
      const msg = normalizeClientError(err);
      if (String(msg).toLowerCase().includes("email non verifie")) {
        openVerificationFlow(
          email,
          "Ce compte n'est pas encore verifie. Le code a ete envoye au client par email. Entre ce code ici ou clique sur 'Resend code'."
        );
        setIsLoginSubmitting(false);
        return;
      }
      if (String(msg).toLowerCase().includes("identifiants invalides")) {
        removeRememberedEmail(email);
        setRememberedEmails(loadRememberedEmails());
        setLoginRememberMe(false);
        setStatus('Identifiants invalides. Si besoin, utilise "Mot de passe oublie ?".');
      } else {
        setStatus(msg);
      }
    } finally {
      setIsLoginSubmitting(false);
    }
  }

  async function verifyLoginCode(e) {
    e.preventDefault();
    setStatus("");
    try {
      const res = await apiFetch("/auth/login/verify-code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: loginEmailPending.trim(),
          code: loginCode.trim()
        })
      });
      const data = await readApiPayload(res);
      if (!res.ok) {
        throw new Error(extractApiError(data, "Erreur de verification du code", res.status));
      }

      const accessToken = String(data.access_token || "");
      if (loginRememberMe) {
        saveStoredToken(accessToken);
        saveRememberedEmail(loginEmailPending.trim());
        setRememberedEmails(loadRememberedEmails());
      } else {
        clearStoredToken();
      }
      setToken(data.access_token);
      setMe(data.user);
      setStatus("Connexion reussie.");
      setScreen("chat");
      setTab("therapy");
      setMessages([]);
      setChatInput("");
      llmSessionIdRef.current = createClientSessionId();
      resetSessionWelcome(data?.user?.username);
      setActiveHistoryId("");
      setLoginHumanCheck(false);
    } catch (err) {
      setStatus(normalizeClientError(err));
    }
  }

  async function forgotPassword(e) {
    e.preventDefault();
    if (isForgotSubmitting) return;
    setStatus("");
    setIsForgotSubmitting(true);
    try {
      const res = await apiFetch("/auth/forgot-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: forgotPasswordEmail.trim()
        })
      });
      const data = await readApiPayload(res);
      if (!res.ok) {
        throw new Error(extractApiError(data, "Erreur de mot de passe oublie", res.status));
      }

      setResetPasswordCode("");
      setStatus(
        buildDeliveryStatus(
          data,
          "Si un compte existe avec cette adresse email, un code de reinitialisation a ete envoye."
        )
      );
      setTab("reset-password");
    } catch (err) {
      setStatus(normalizeClientError(err));
    } finally {
      setIsForgotSubmitting(false);
    }
  }

  async function resetPassword(e) {
    e.preventDefault();
    if (isResetSubmitting) return;
    setStatus("");
    setIsResetSubmitting(true);
    try {
      const res = await apiFetch("/auth/reset-password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: forgotPasswordEmail.trim(),
          code: resetPasswordCode.trim(),
          newPassword: resetPasswordNew.trim(),
          confirmPassword: resetPasswordConfirm.trim()
        })
      });
      const data = await readApiPayload(res);
      if (!res.ok) {
        throw new Error(extractApiError(data, "Erreur de reinitialisation du mot de passe", res.status));
      }

      setStatus("Mot de passe reinitialise avec succes. Vous pouvez maintenant vous connecter.");
      setTab("login");
      setForgotPasswordEmail("");
      setResetPasswordCode("");
      setResetPasswordNew("");
      setResetPasswordConfirm("");
    } catch (err) {
      setStatus(normalizeClientError(err));
    } finally {
      setIsResetSubmitting(false);
    }
  }

  async function sendTextTherapy(e) {
    e?.preventDefault?.();
    if (!chatInput.trim() || isSending) return;
    setStatus("Therapist is preparing a reply... first response can take up to 2-4 minutes.");
    setIsSending(true);
    setHasComposerInteracted(true);

    const prompt = chatInput.trim();
    const history = buildModelHistory([...messages, { role: "user", content: prompt }]);
    addMessage("user", prompt);
    setChatInput("");

    try {
      const res = await apiFetch("/session/text-auth", {
        method: "POST",
        headers: { ...headers, "Content-Type": "application/json" },
        body: JSON.stringify({
          text: prompt,
          session_id: llmSessionIdRef.current,
          history
        })
      });
      const data = await readApiPayload(res);
      if (!res.ok) {
        throw new Error(extractApiError(data, "Text session error", res.status));
      }

      const emotion = data.emotion || "neutral";
      const reply = data.reply || "";
      addMessage("assistant", reply || "I am here with you.", {
        emotion,
        ttsAudioBase64: data.tts_audio_base64 || "",
        ttsAudioMime: data.tts_audio_mime || "audio/wav",
        ttsEngine: data.tts_engine || ""
      });
      setStatus("");
      setActiveHistoryId("");
      loadHistory(true);
    } catch (err) {
      const message = String(err.message || err);
      setStatus(message);
      addMessage("system", `Error: ${message}`);
    } finally {
      setIsSending(false);
    }
  }

  function handleComposerKeyDown(e) {
    if (e.key !== "Enter" || e.nativeEvent?.isComposing) return;
    e.preventDefault();
    setHasComposerInteracted(true);
    sendTextTherapy();
  }

  async function uploadAudioFile(file) {
    if (!file || file.size === 0) {
      setStatus("No valid audio file.");
      return;
    }

    const payload = new FormData();
    payload.append("audio", file);
    payload.append("session_id", llmSessionIdRef.current);
    addMessage("system", "Transcribing...");

    try {
      setIsVoiceSending(true);
      const res = await apiFetch("/session/audio-upload", {
        method: "POST",
        headers,
        body: payload
      });
      const data = await readApiPayload(res);
      if (!res.ok) {
        throw new Error(extractApiError(data, "Audio upload error", res.status));
      }

      const transcript = (data.transcript || "").trim();
      if (transcript) {
        addMessage("user", transcript);
      } else {
        addMessage("system", "Empty transcription.");
      }

      addAssistantMessage(
        data.reply || "I am here with you.",
        {
          emotion: data.emotion || "neutral",
          ttsAudioBase64: data.tts_audio_base64 || "",
          ttsAudioMime: data.tts_audio_mime || "audio/wav",
          ttsEngine: data.tts_engine || ""
        },
        true
      );
      setVoiceOpen(false);
      setActiveHistoryId("");
      loadHistory(true);
    } catch (err) {
      const message = String(err.message || err);
      setStatus(message);
      addMessage("system", `Audio error: ${message}`);
    } finally {
      setIsVoiceSending(false);
      setIsRecording(false);
    }
  }

  async function uploadVoiceBlob(blob) {
    if (!blob || blob.size === 0) {
      setStatus("No recorded audio.");
      return;
    }
    const file = new File([blob], `voice-${Date.now()}.webm`, {
      type: blob.type || "audio/webm"
    });
    await uploadAudioFile(file);
  }

  async function startVoiceRecording() {
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setStatus("Voice recording is not supported on this browser.");
      return;
    }
    try {
      setStatus("");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      voiceChunksRef.current = [];
      setVoiceSeconds(0);
      setIsRecording(true);

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          voiceChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        if (voiceTimerRef.current) {
          clearInterval(voiceTimerRef.current);
          voiceTimerRef.current = null;
        }
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop());
          streamRef.current = null;
        }
        const blob = new Blob(voiceChunksRef.current, { type: "audio/webm" });
        await uploadVoiceBlob(blob);
      };

      voiceTimerRef.current = setInterval(() => {
        setVoiceSeconds((prev) => {
          const next = prev + 1;
          if (
            next >= MAX_VOICE_SECONDS &&
            mediaRecorderRef.current &&
            mediaRecorderRef.current.state === "recording"
          ) {
            mediaRecorderRef.current.stop();
            mediaRecorderRef.current = null;
          }
          return next;
        });
      }, 1000);

      recorder.start();
    } catch (err) {
      const message = String(err.message || err);
      setStatus(message);
      addMessage("system", `Microphone error: ${message}`);
      setIsRecording(false);
    }
  }

  function stopVoiceRecording() {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
    }
  }

  function closeVoicePanel() {
    if (isRecording || isVoiceSending) return;
    setVoiceOpen(false);
  }

  function logout() {
    stopSpeech();
    clearStoredToken();
    setToken("");
    setMe(null);
    setStatus("");
    setIsAuthBootstrapping(false);
    setScreen("chat");
    setTab("login");
    setLoginRememberMe(false);
    setLoginHumanCheck(false);
    setLoginPasswordVisible(false);
    setLoginEmail("");
    setMessages([]);
    setChatInput("");
    resetSessionWelcome("ami");
    setHistoryItems([]);
    setHistoryLoading(false);
    setActiveHistoryId("");
    setOpenHistoryMenuId("");
    setIsUploadingPhoto(false);
    setIsCameraOpen(false);
    setProfileUsername("");
    setProfileOldPassword("");
    setProfileNewPassword("");
    setProfileOldPasswordVisible(false);
    setProfileNewPasswordVisible(false);
    setIsProfileSaving(false);
    setIsPasswordSaving(false);
    llmSessionIdRef.current = createClientSessionId();
    stopProfileCameraStream();
  }

  async function loadHistory(silent = false) {
    if (!token) return;
    setHistoryLoading(true);
    if (!silent) {
      setStatus("");
    }
    try {
      const res = await apiFetch("/session/history", { headers });
      const data = await readApiPayload(res);
      if (!res.ok) {
        throw new Error(extractApiError(data, "History error", res.status));
      }
      setHistoryItems(sortHistoryItems(Array.isArray(data.items) ? data.items : []));
    } catch (err) {
      if (!silent) {
        setStatus(normalizeClientError(err));
      }
    } finally {
      setHistoryLoading(false);
    }
  }

  function startNewChat() {
    stopSpeech();
    setMessages([]);
    setChatInput("");
    resetSessionWelcome(me?.username);
    setActiveHistoryId("");
    setOpenHistoryMenuId("");
    setScreen("chat");
    setStatus("New chat started.");
    llmSessionIdRef.current = createClientSessionId();
  }

  function openHistoryEntry(item) {
    if (!item) return;
    stopSpeech();
    setMessages(historyItemToMessages(item));
    setChatInput("");
    setHasComposerInteracted(true);
    setActiveHistoryId(String(item.id || ""));
    setOpenHistoryMenuId("");
    setScreen("chat");
    const restoredSessionId = String(item?.session_id || "").trim();
    llmSessionIdRef.current = restoredSessionId || `history-${String(item.id || "default")}`;
  }

  async function renameHistoryItem(item) {
    const id = String(item?.id || "");
    if (!id) return;
    const input = window.prompt(
      `New chat name (current: ${historyTitle(item)})`,
      ""
    );
    if (input == null) return;
    const title = input.trim();
    if (!title) {
      setStatus("Chat name cannot be empty.");
      return;
    }
    try {
      const res = await apiFetch(`/session/history/${id}/rename`, {
        method: "PATCH",
        headers: { ...headers, "Content-Type": "application/json" },
        body: JSON.stringify({ title })
      });
      const data = await readApiPayload(res);
      if (!res.ok) {
        throw new Error(extractApiError(data, "Rename error", res.status));
      }
      const updated = data?.item || {};
      setHistoryItems((prev) =>
        sortHistoryItems(prev.map((entry) => (String(entry.id) === id ? { ...entry, ...updated } : entry)))
      );
      setStatus("Chat renamed.");
    } catch (err) {
      setStatus(normalizeClientError(err));
    } finally {
      setOpenHistoryMenuId("");
    }
  }

  async function togglePinHistoryItem(item) {
    const id = String(item?.id || "");
    if (!id) return;
    const pinned = !Boolean(item?.pinned);
    try {
      const res = await apiFetch(`/session/history/${id}/pin`, {
        method: "PATCH",
        headers: { ...headers, "Content-Type": "application/json" },
        body: JSON.stringify({ pinned })
      });
      const data = await readApiPayload(res);
      if (!res.ok) {
        throw new Error(extractApiError(data, "Pin error", res.status));
      }
      const updated = data?.item || { ...item, pinned };
      setHistoryItems((prev) =>
        sortHistoryItems(prev.map((entry) => (String(entry.id) === id ? { ...entry, ...updated } : entry)))
      );
      setStatus(pinned ? "Chat pinned." : "Chat unpinned.");
    } catch (err) {
      setStatus(normalizeClientError(err));
    } finally {
      setOpenHistoryMenuId("");
    }
  }

  async function deleteHistoryItem(item) {
    const id = String(item?.id || "");
    if (!id) return;
    const confirmed = window.confirm(`Delete chat "${historyTitle(item)}"?`);
    if (!confirmed) return;
    try {
      const res = await apiFetch(`/session/history/${id}`, {
        method: "DELETE",
        headers
      });
      const data = await readApiPayload(res);
      if (!res.ok) {
        throw new Error(extractApiError(data, "Delete error", res.status));
      }
      setHistoryItems((prev) => prev.filter((entry) => String(entry.id) !== id));
      if (activeHistoryId === id) {
        setMessages([]);
        setChatInput("");
        resetSessionWelcome(me?.username);
        setActiveHistoryId("");
      }
      setStatus("Chat deleted.");
    } catch (err) {
      setStatus(normalizeClientError(err));
    } finally {
      setOpenHistoryMenuId("");
    }
  }

  if (isAuthBootstrapping) {
    return (
      <div className="auth-shell">
        <main className="auth-panel">
          <header className="auth-header">
            {isAuthBackVisible && (
              <button type="button" className="auth-back-arrow" onClick={backToLogin} aria-label="Retour">
                в†ђ
              </button>
            )}
            <h1>Therapist AI</h1>
            <p>Restoring your session...</p>
          </header>
        </main>
      </div>
    );
  }

  if (!token) {
    return (
      <div className="auth-shell">
        <main className="auth-panel">
          <header className="auth-header">
            {isAuthBackVisible && (
              <button type="button" className="auth-back-arrow" onClick={backToLogin} aria-label="Retour">
                в†ђ
              </button>
            )}
            <h1>Therapist AI</h1>
            <p>Sign in to access your conversation space.</p>
          </header>

          <nav className="auth-tabs">
            <button
              className={tab === "login" ? "active" : ""}
              onClick={() => switchAuthTab("login")}
            >
              Connexion
            </button>
            <button
              className={tab === "register" ? "active" : ""}
              onClick={() => switchAuthTab("register")}
            >
              Inscription
            </button>
          </nav>

          {status && <span className="status-chip">{status}</span>}

          {tab === "register" && (
            <form className="auth-card" onSubmit={registerUser}>
              <h3>Creer un compte</h3>
              <input name="email" type="email" placeholder="Email" required />
              <input
                name="confirmEmail"
                type="email"
                placeholder="Confirmer l'email"
                required
              />
              <input name="username" type="text" placeholder="Nom d'utilisateur" required />
              <select name="sex" required defaultValue="">
                <option value="" disabled>
                  Sexe
                </option>
                <option value="M">Homme</option>
                <option value="F">Femme</option>
                <option value="Other">Autre</option>
              </select>
              <input
                name="age"
                type="number"
                min="10"
                max="120"
                placeholder="Age"
                required
              />
              <input name="password" type="password" placeholder="Mot de passe" required />
              <input
                name="confirmPassword"
                type="password"
                placeholder="Confirmer le mot de passe"
                required
              />
              <button type="submit">Creer un compte</button>
            </form>
          )}

          {tab === "verify" && (
            <form className="auth-card" onSubmit={verifyEmail}>
              <h3>Verification de l'email</h3>
              <input
                value={verifyEmailValue}
                onChange={(e) => setVerifyEmailValue(e.target.value)}
                placeholder="Email"
                type="email"
                required
              />
              <input
                value={verifyCode}
                onChange={(e) => setVerifyCode(e.target.value)}
                placeholder="Code recu par email"
                required
              />
              <button type="submit">Verifier</button>
              <button
                type="button"
                className="auth-secondary-btn"
                onClick={resendVerificationCode}
                disabled={isResendingVerification || !verifyEmailValue.trim()}
              >
                {isResendingVerification ? "Envoi..." : "Renvoyer le code"}
              </button>
              <p className="auth-info">
                Le code doit etre recu par email. Si rien n'arrive, verifiez le SMTP et le dossier spam.
              </p>
            </form>
          )}

          {tab === "login" && (
            <form className="auth-card" onSubmit={login}>
              <h3>Connexion</h3>
              <input
                name="email"
                type="email"
                placeholder="Email"
                value={loginEmail}
                onChange={(e) => setLoginEmail(e.target.value)}
                required
              />
              <div className="auth-password-field">
                <button
                  type="button"
                  className="password-visibility-btn"
                    aria-label={loginPasswordVisible ? "Masquer le mot de passe" : "Afficher le mot de passe"}
                  onClick={() => setLoginPasswordVisible((prev) => !prev)}
                >
                  {"\u{1F441}"}
                </button>
                <input
                  name="password"
                  type={loginPasswordVisible ? "text" : "password"}
                    placeholder="Mot de passe"
                  required
                />
              </div>
              <label className="auth-check">
                <input
                  name="humanCheck"
                  type="checkbox"
                  checked={loginHumanCheck}
                  onChange={(e) => setLoginHumanCheck(e.target.checked)}
                  required
                />
                <span>Je ne suis pas un robot</span>
              </label>
              {!isLoginEmailRemembered && (
                <label className="auth-check">
                  <input
                    name="rememberMe"
                    type="checkbox"
                    checked={loginRememberMe}
                    onChange={(e) => setLoginRememberMe(e.target.checked)}
                  />
                    <span>Se souvenir de moi sur cet appareil</span>
                </label>
              )}
              {isLoginEmailRemembered && (
                <p className="auth-info">
                  Cet email est deja approuve sur cet appareil. Aucun code de connexion n'est requis.
                </p>
              )}
              <button type="submit" disabled={isLoginSubmitting}>
                {isLoginSubmitting
                  ? "Connexion en cours..."
                  : "Se connecter"}
              </button>
              <button
                type="button"
                className="forgot-password-link"
                onClick={() => switchAuthTab("forgot-password")}
              >
                Mot de passe oublie ?
              </button>
            </form>
          )}

          {tab === "login-code" && (
            <form className="auth-card" onSubmit={verifyLoginCode}>
              <h3>Code de connexion</h3>
              <input
                value={loginEmailPending}
                onChange={(e) => setLoginEmailPending(e.target.value)}
                placeholder="Email"
                type="email"
                required
              />
              <input
                value={loginCode}
                onChange={(e) => setLoginCode(e.target.value)}
                placeholder="Code recu par email"
                required
              />
              <button type="submit">Valider le code</button>
              <p className="auth-info">
                Le code de connexion doit etre envoye par email au client.
              </p>
            </form>
          )}

          {tab === "forgot-password" && (
            <form className="auth-card" onSubmit={forgotPassword}>
              <h3>Mot de passe oublie</h3>
              <input
                value={forgotPasswordEmail}
                onChange={(e) => setForgotPasswordEmail(e.target.value)}
                placeholder="Email"
                type="email"
                required
              />
              <button type="submit" disabled={isForgotSubmitting}>
                {isForgotSubmitting
                  ? "Envoi en cours..."
                  : "Envoyer le code"}
              </button>
              <p className="auth-info">
                Si vous ne recevez pas le code par mail, verifiez la configuration SMTP puis le dossier spam.
              </p>
            </form>
          )}

          {tab === "reset-password" && (
            <form className="auth-card" onSubmit={resetPassword}>
              <h3>Reinitialiser le mot de passe</h3>
              <input
                value={forgotPasswordEmail}
                onChange={(e) => setForgotPasswordEmail(e.target.value)}
                placeholder="Email"
                type="email"
                required
              />
              <input
                value={resetPasswordCode}
                onChange={(e) => setResetPasswordCode(e.target.value)}
                placeholder="Code de reinitialisation"
                required
              />
              <input
                value={resetPasswordNew}
                onChange={(e) => setResetPasswordNew(e.target.value)}
                placeholder="Nouveau mot de passe"
                type="password"
                required
              />
              <input
                value={resetPasswordConfirm}
                onChange={(e) => setResetPasswordConfirm(e.target.value)}
                placeholder="Confirmer le mot de passe"
                type="password"
                required
              />
              <button type="submit" disabled={isResetSubmitting}>
                {isResetSubmitting
                  ? "Reinitialisation en cours..."
                  : "Reinitialiser"}
              </button>
            </form>
          )}
        </main>
      </div>
    );
  }

  const userInitial = (me?.username || "U").slice(0, 1).toUpperCase();
  const emotionCounts = messages.reduce((acc, msg) => {
    if (msg.role !== "assistant") return acc;
    const key = String(msg?.emotion || "neutral").trim().toLowerCase();
    acc[key] = (acc[key] || 0) + 1;
    return acc;
  }, {});
  const topEmotions = Object.entries(emotionCounts).slice(0, 6);
  const quickMoods = [
    "\u{1F642}",
    "\u{1F610}",
    "\u{1F622}",
    "\u{1F61F}",
    "\u{1F621}",
    "\u{1F976}"
  ];
  const primaryEmotion = topEmotions[0]?.[0] || "neutral";
  const assistantCount = messages.filter((msg) => msg.role === "assistant").length;
  const isProfileScreen = screen === "profile";
  const profileJoinedAt = formatHistoryDate(me?.created_at);
  const showChatWelcome =
    messages.length === 0 &&
    !hasComposerInteracted &&
    !chatInput.trim() &&
    screen === "chat";

  return (
    <div className="app-shell modern-shell">
      <aside className="sidebar sidebar-modern history-sidebar">
        <div className="brand">
          <span className="brand-dot">AI</span>
          <div>
            <h1>Therapist AI</h1>
            <p>Hello {me?.username || "client"}</p>
          </div>
        </div>

        <button type="button" className="new-chat-btn" onClick={startNewChat}>
          + New chat
        </button>

        <section className="history-panel">
          <h2>History</h2>
          <div className="history-list">
            {historyLoading && <p className="history-empty">Loading...</p>}
            {!historyLoading && historyItems.length === 0 && (
              <p className="history-empty">No saved conversations.</p>
            )}
            {!historyLoading &&
              historyItems.map((item) => (
                <article
                  key={String(item.id)}
                  className={`history-item ${activeHistoryId === String(item.id) ? "active" : ""
                    } ${openHistoryMenuId === String(item.id) ? "menu-open" : ""}`}
                >
                  <button type="button" className="history-open-btn" onClick={() => openHistoryEntry(item)}>
                    <div className="history-row-top">
                      <strong className="history-title">{historyTitle(item)}</strong>
                    </div>
                    <span className="history-preview">{historyPreview(item)}</span>
                    <small className="history-meta">
                      {item?.channel || "text"} {formatHistoryDate(item?.created_at)}
                    </small>
                  </button>

                  <div className="history-item-actions">
                    {item?.pinned && (
                      <span className="history-pin-icon" title="Pinned" aria-label="Pinned" />
                    )}
                    <button
                      type="button"
                      className="history-menu-trigger"
                      aria-label="Chat options"
                      onClick={(e) => {
                        e.stopPropagation();
                        setOpenHistoryMenuId((prev) => (prev === String(item.id) ? "" : String(item.id)));
                      }}
                    >
                      ...
                    </button>

                    {openHistoryMenuId === String(item.id) && (
                      <div
                        className="history-menu"
                        onClick={(e) => e.stopPropagation()}
                        onMouseDown={(e) => e.stopPropagation()}
                      >
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            renameHistoryItem(item);
                          }}
                        >
                          Rename chat
                        </button>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            togglePinHistoryItem(item);
                          }}
                        >
                          {item?.pinned ? "Unpin chat" : "Pin chat"}
                        </button>
                        <button
                          type="button"
                          className="danger"
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteHistoryItem(item);
                          }}
                        >
                          Delete chat
                        </button>
                      </div>
                    )}
                  </div>
                </article>
              ))}
          </div>
        </section>

        <section className="profile-card">
          <button
            type="button"
            className={`profile-open-link ${isProfileScreen ? "active" : ""}`}
            onClick={() => setScreen("profile")}
          >
            <span className="profile-open-name">{me?.username || "Account"}</span>
            <span className="profile-open-hint">Ouvrir le profil</span>
          </button>
          <p>{me?.email}</p>
          <button onClick={logout}>Se deconnecter</button>
        </section>
      </aside>

      <main className="main-panel">
        <header className="topbar modern-topbar">
          <div>
            <h2>{isProfileScreen ? "Votre profil" : `Bonjour ${me?.username || "client"}`}</h2>
            <p>
              {isProfileScreen
                ? "Gerez votre nom, votre mot de passe et votre photo de profil."
                : "Comment vous sentez-vous aujourd'hui ?"}
            </p>
            {!isProfileScreen && (
              <div className="mood-row">
                {quickMoods.map((mood) => (
                  <button key={mood} type="button" className="mood-btn">
                    {mood}
                  </button>
                ))}
              </div>
            )}
          </div>
          <div className="top-right">
            <div className="top-icons">
              <button type="button" className="top-icon" onClick={() => setScreen("chat")}>
                Chat
              </button>
              <button type="button" className="top-icon" onClick={() => setScreen("emotions")}>
                Mood
              </button>
              <button type="button" className="top-icon" onClick={() => setScreen("profile")}>
                Profile
              </button>
              <div className="top-user-badge top-user-badge-static" title="Profile photo">
                {me?.photo_path ? (
                  <img src={me.photo_path} alt="Profile" className="top-user-photo" />
                ) : (
                  <span>{userInitial}</span>
                )}
              </div>
            </div>
            {status && <span className="status-chip">{status}</span>}
          </div>
        </header>

        <input
          ref={profileFileInputRef}
          className="profile-file-input"
          type="file"
          accept="image/jpeg,image/png,image/webp,image/gif"
          onChange={handleProfilePhotoChange}
          disabled={isUploadingPhoto}
        />

        {screen === "dashboard" && (
          <section className="dashboard-grid">
            <button className="dash-card voice" onClick={() => setScreen("voice")}>
              <span className="dash-icon">MIC</span>
              <h3>Speak</h3>
              <p>Voice session</p>
              <strong>Start</strong>
            </button>
            <button className="dash-card chat" onClick={() => setScreen("chat")}>
              <span className="dash-icon">MSG</span>
              <h3>Send a message</h3>
              <p>Chat with Therapist AI</p>
              <strong>Chat</strong>
            </button>
            <button className="dash-card mood" onClick={() => setScreen("emotions")}>
              <span className="dash-icon">MOOD</span>
                <h3>Suivre mon humeur</h3>
                <p>Voir l evolution des emotions detectees</p>
                <strong>Voir</strong>
            </button>
          </section>
        )}

        {screen === "chat" && (
          <section className="chat-layout">
            <div className="chat-column">
              <section className={`chat-thread ${showChatWelcome ? "chat-thread-empty" : ""}`}>
                {showChatWelcome && (
                  <article className="chat-welcome-card">
                    <h3>{chatWelcome.title}</h3>
                    <p>{chatWelcome.subtitle}</p>
                  </article>
                )}
                {messages.map((msg, index) => {
                  const assistantEmotion =
                    msg.role === "assistant" ? normalizeEmotionKey(msg?.emotion) : "";
                  return (
                    <article key={`${msg.role}-${index}`} className={`msg ${msg.role}`}>
                      <div className="msg-head">
                        <div className="msg-meta">
                          <span className="msg-role">
                            {formatMessageRole(msg.role, me?.username)}
                          </span>
                          {msg.role === "assistant" && (
                            <span className={`msg-emotion msg-emotion-${assistantEmotion}`}>
                              {formatEmotionLabel(assistantEmotion)}
                            </span>
                          )}
                        </div>
                        {msg.role === "assistant" && (
                          <button
                            type="button"
                            className="speak-btn"
                            onClick={() => speakMessage(msg, index)}
                            title="Listen to reply"
                            aria-label={
                              speakingIndex === index
                                ? "Stop voice playback"
                                : "Listen to reply audio"
                            }
                          >
                            {speakingIndex === index ? "Stop" : "\u{1F3A4}"}
                          </button>
                        )}
                      </div>
                      <p>{msg.content}</p>
                    </article>
                  );
                })}
              </section>

              <section className="composer-area">
                <form className="composer" onSubmit={sendTextTherapy}>
                  <input
                    name="text"
                    type="text"
                    className="composer-input-line"
                    value={chatInput}
                    onChange={(e) => {
                      if (e.target.value) {
                        setHasComposerInteracted(true);
                      }
                      setChatInput(e.target.value);
                    }}
                    onFocus={() => setHasComposerInteracted(true)}
                    onKeyDown={handleComposerKeyDown}
                    placeholder="Partagez ce que vous ressentez..."
                    autoComplete="off"
                  />
                  <div className="composer-right-tools">
                    <span className="composer-right-dot" aria-hidden="true" />
                    <button
                      type="button"
                      className="voice-icon-btn"
                      onClick={() => setVoiceOpen(true)}
                      disabled={isVoiceSending}
                      title="Mode vocal"
                      aria-label="Ouvrir le mode vocal"
                    >
                      <svg
                        className="voice-mic-icon"
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                        aria-hidden="true"
                      >
                        <rect x="9" y="2.5" width="6" height="11" rx="3" stroke="currentColor" strokeWidth="2" />
                        <path d="M5 10.5C5 14.09 7.91 17 11.5 17H12.5C16.09 17 19 14.09 19 10.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                        <path d="M12 17V21.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                        <path d="M8.5 21.5H15.5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                      </svg>
                    </button>
                  </div>
                </form>
              </section>
            </div>

            <aside className="chat-side">
              <section className="side-card">
                <h4>Etat actuel</h4>
                <p>
                  Emotion dominante : <b>{primaryEmotion}</b>
                </p>
                <p>
                  Reponses de l'IA : <b>{assistantCount}</b>
                </p>
              </section>
              <section className="side-card">
                <h4>Mode vocal</h4>
                <p>
                  {isRecording
                    ? `Listening... auto-stop ${MAX_VOICE_SECONDS}s`
                    : "Parlez librement, puis arretez."}
                </p>
                {!isRecording ? (
                  <button onClick={startVoiceRecording} disabled={isVoiceSending}>
                    Demarrer
                  </button>
                ) : (
                  <button onClick={stopVoiceRecording}>Arreter</button>
                )}
              </section>
            </aside>
          </section>
        )}

        {screen === "voice" && (
          <section className="voice-page">
            <div className={`voice-robot big ${isRecording ? "live" : ""}`}>BOT</div>
            <p>
              {isRecording
                ? `I am listening... auto-stop ${MAX_VOICE_SECONDS}s`
                : "Cliquez pour demarrer la session vocale"}
            </p>
            <span className="voice-timer">{formatDuration(voiceSeconds)}</span>
            <div className="voice-actions">
              {!isRecording ? (
                <button onClick={startVoiceRecording} disabled={isVoiceSending}>
                  Demarrer
                </button>
              ) : (
                <button onClick={stopVoiceRecording}>Arreter</button>
              )}
            </div>
          </section>
        )}

        {screen === "profile" && (
          <section className="profile-page-modern">
            <article className="profile-hero">
              <div className="profile-avatar-xl">
                {me?.photo_path ? (
                  <img src={me.photo_path} alt="Profile" className="top-user-photo" />
                ) : (
                  <span>{userInitial}</span>
                )}
              </div>
              <div className="profile-hero-copy">
                <h3>{me?.username || "Account"}</h3>
                <p>{me?.email || "No email"}</p>
                {profileJoinedAt && <small>Member since {profileJoinedAt}</small>}
              </div>
              <div className="profile-hero-actions">
                <button
                  type="button"
                  onClick={openProfileCamera}
                  disabled={isUploadingPhoto || isCameraStarting}
                >
                  Prendre une photo
                </button>
                <button
                  type="button"
                  onClick={triggerLocalPhotoImport}
                  disabled={isUploadingPhoto}
                >
                  Importer une photo
                </button>
                <button
                  type="button"
                  className="danger-btn"
                  onClick={deleteProfilePhoto}
                  disabled={isUploadingPhoto || !me?.photo_path}
                >
                  Supprimer la photo
                </button>
              </div>
            </article>

            <div className="profile-settings-grid">
              <form className="profile-settings-card" onSubmit={saveProfileUsername}>
                <h4>Nom d'utilisateur</h4>
                <p>Choisissez comment votre nom apparait dans votre espace.</p>
                <input
                  type="text"
                  value={profileUsername}
                  onChange={(e) => setProfileUsername(e.target.value)}
                  placeholder="Nom d'utilisateur"
                  minLength={3}
                  maxLength={32}
                  required
                />
                <button type="submit" disabled={isProfileSaving}>
                  {isProfileSaving ? "Enregistrement..." : "Enregistrer le nom"}
                </button>
              </form>

              <form className="profile-settings-card" onSubmit={saveProfilePassword}>
                <h4>Mot de passe</h4>
                <p>Modifiez votre mot de passe en saisissant d'abord le mot de passe actuel.</p>
                <div className="auth-password-field profile-password-field">
                  <button
                    type="button"
                    className="password-visibility-btn"
                    aria-label={profileOldPasswordVisible ? "Masquer l'ancien mot de passe" : "Afficher l'ancien mot de passe"}
                    onClick={() => setProfileOldPasswordVisible((prev) => !prev)}
                  >
                    {"\u{1F441}"}
                  </button>
                  <input
                    type={profileOldPasswordVisible ? "text" : "password"}
                    value={profileOldPassword}
                    onChange={(e) => setProfileOldPassword(e.target.value)}
                    placeholder="Ancien mot de passe"
                    autoComplete="current-password"
                    required
                  />
                </div>
                <div className="auth-password-field profile-password-field">
                  <button
                    type="button"
                    className="password-visibility-btn"
                    aria-label={profileNewPasswordVisible ? "Masquer le nouveau mot de passe" : "Afficher le nouveau mot de passe"}
                    onClick={() => setProfileNewPasswordVisible((prev) => !prev)}
                  >
                    {"\u{1F441}"}
                  </button>
                  <input
                    type={profileNewPasswordVisible ? "text" : "password"}
                    value={profileNewPassword}
                    onChange={(e) => setProfileNewPassword(e.target.value)}
                    placeholder="Nouveau mot de passe"
                    autoComplete="new-password"
                    minLength={6}
                    required
                  />
                </div>
                <button type="submit" disabled={isPasswordSaving}>
                  {isPasswordSaving ? "Mise a jour..." : "Mettre a jour le mot de passe"}
                </button>
              </form>

              <section className="profile-settings-card profile-danger-card">
                <h4>Compte</h4>
                <p>Supprimez definitivement votre compte et vos informations associees.</p>
                <button type="button" className="danger-btn" onClick={deleteAccount}>
                  Supprimer le compte
                </button>
              </section>
            </div>
          </section>
        )}

        {screen === "emotions" && (
          <section className="emotion-page">
            <h3>Detected history</h3>
            <div className="emotion-bars">
              {(topEmotions.length ? topEmotions : [["neutral", 0]]).map(([emotion, count]) => (
                <div key={emotion} className="emotion-row">
                  <span>{emotion}</span>
                  <div className="emotion-bar">
                    <i style={{ width: `${Math.min(100, Number(count) * 20)}%` }} />
                  </div>
                  <b>{count}</b>
                </div>
              ))}
            </div>
          </section>
        )}

        <nav className="app-dock">
          <button onClick={() => setScreen("dashboard")}>Home</button>
          <button onClick={() => setScreen("chat")}>Chat</button>
          <button onClick={() => setScreen("voice")}>Voice</button>
          <button onClick={() => setScreen("emotions")}>Emotions</button>
          <button onClick={() => setScreen("profile")}>Profile</button>
        </nav>
      </main>

      {voiceOpen && (
        <div className="voice-overlay" onClick={closeVoicePanel}>
          <section className="voice-modal" onClick={(e) => e.stopPropagation()}>
            <div className={`voice-robot ${isRecording ? "live" : ""}`}>BOT</div>
            <h3>Mode vocal</h3>
            <p>
              {isVoiceSending
                ? "Analyse en cours..."
                : isRecording
                  ? `I am listening... auto-stop ${MAX_VOICE_SECONDS}s`
                  : "Cliquez sur Demarrer pour parler."}
            </p>
            <span className="voice-timer">{formatDuration(voiceSeconds)}</span>
            <div className="voice-actions">
              {!isRecording ? (
                <button onClick={startVoiceRecording} disabled={isVoiceSending}>
                  Demarrer
                </button>
              ) : (
                <button onClick={stopVoiceRecording}>Arreter</button>
              )}
              <button onClick={closeVoicePanel} disabled={isRecording || isVoiceSending}>
                Fermer
              </button>
            </div>
          </section>
        </div>
      )}

      {isCameraOpen && (
        <div className="camera-overlay" onClick={closeProfileCamera}>
          <section className="camera-modal" onClick={(e) => e.stopPropagation()}>
            <h3>Camera en direct</h3>
            <video ref={profileCameraVideoRef} className="camera-feed" autoPlay playsInline muted />
            <div className="camera-actions">
              <button
                type="button"
                onClick={captureProfilePhotoFromCamera}
                disabled={isUploadingPhoto}
              >
                Capturer
              </button>
              <button
                type="button"
                onClick={closeProfileCamera}
                disabled={isUploadingPhoto}
              >
                Fermer
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

export default App;

