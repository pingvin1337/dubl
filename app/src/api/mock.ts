/**
 * Мок-движок: работает, когда настоящий движок недоступен (плашка «Демо-режим»).
 * Те же методы, что у HttpEngine. Задачи идут по таймерам, реплики — из mockData.ts.
 * Видео нет (URL = null), плеер рисует канвас-заглушку с таймкодом.
 */
import { EngineError, type Engine } from "./engine";
import { buildAnalysis, FRIDGE_AT, planFor, RU_BY_EN, SPEAKER_DEFS } from "./mockData";
import { defaultMix, defaultTrack } from "./presets";
import { applyPatches, parseCommand, takeRemark, takeReport } from "../lib/director";
import { estimateDuration, guessLang } from "../lib/estimate";
import { rng } from "../lib/rng";
import { selectByMode } from "../lib/snap";
import { clamp } from "../lib/format";
import { decodeWav, wavBlob } from "../audio/wav";
import { peakDb, rmsDb, speechBounds } from "../audio/analyze";
import type {
  Analysis, AppSettings, AudioStreamInfo, CheckResult, LatencyResult, MicCheckResult, CreateProjectOptions, DirectorCommandResult, DirectorTakeResult, FitLengthResult, Health, Job,
  JobStep, KeyResult, LLMRequest, LinesMergeResult, MediaInfo, MergeResult, UrlProbe, MixSettings, NameSpeakersResult, Project, ProjectSummary, ProjectWithJob, ProxySource,
  ReanalyzeOptions, RenderProjectRequest, Scene, SceneCreate, SceneLine, SceneWithJob, ScenePatch, SnapRequest, SnapResult, Take, TakePatch,
  TakeUploadMeta, TranslateResult, AnalysisX, ProjectSummaryX,
} from "./types";

export interface MockOptions {
  /** во сколько раз ускорить таймеры задач (тесты) */
  speed?: number;
  /** положить демо-проекты (по умолчанию да) */
  seed?: boolean;
  /** есть ли «нейросети» (как capabilities.ml у настоящего движка); false — разбор падает с error_code=ml_missing */
  ml?: boolean;
}

export const ML_MISSING_TEXT = "Не установлены нейросети — без них нельзя распознавать речь и отделять голоса. Скачай их один раз (~1,3 ГБ).";

const ANALYZE_STEPS: [string, string][] = [
  ["probe", "Чтение файла"], ["audio", "Звук из видео"], ["subtitles", "Субтитры из файла"],
  ["transcribe", "Текст и тайминги"], ["diarize", "Кто говорит"], ["shots", "Склейки кадров"],
];
const PREPARE_STEPS: [string, string][] = [
  ["cut", "Вырезка сцены"], ["audio", "Звук сцены"], ["separate", "Голоса и фон"], ["speakers", "Голос каждого персонажа"],
];
const RENDER_STEPS: [string, string][] = [["mix", "Сведение"], ["master", "Мастеринг"], ["video", "Сборка видео"]];

const iso = (d = new Date()) => d.toISOString();
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const clone = <T,>(x: T): T => structuredClone(x);

export function blobToArrayBuffer(b: Blob): Promise<ArrayBuffer> {
  if (typeof b.arrayBuffer === "function") return b.arrayBuffer();
  return new Promise((res, rej) => {
    const fr = new FileReader();
    fr.onload = () => res(fr.result as ArrayBuffer);
    fr.onerror = () => rej(fr.error);
    fr.readAsArrayBuffer(b);
  });
}

/** «Голос» из шумовых слогов — чтобы у демо-дублей была волна и их можно было послушать. */
export function synthVoice(seconds: number, lead: number, seed: number, sr = 48000): Float32Array {
  const r = rng(seed);
  const out = new Float32Array(Math.floor((seconds + lead + 0.4) * sr));
  const pitch = 110 + r() * 60;
  let t = lead;
  while (t < lead + seconds - 0.1) {
    const syl = 0.12 + r() * 0.12;
    const gap = r() < 0.2 ? 0.14 + r() * 0.1 : 0.02;
    const f = pitch * (0.9 + r() * 0.3);
    const a = Math.floor(t * sr);
    const n = Math.floor(syl * sr);
    for (let i = 0; i < n && a + i < out.length; i++) {
      const env = Math.sin((Math.PI * i) / n) ** 1.5;
      const x = i / sr;
      const v = Math.sin(2 * Math.PI * f * x) + 0.5 * Math.sin(4 * Math.PI * f * x) + 0.25 * Math.sin(6 * Math.PI * f * x) + (r() - 0.5) * 0.25;
      out[a + i] += v * env * 0.22;
    }
    t += syl + gap;
  }
  return out;
}

function fileKind(name: string): "rich" | "plain" {
  return /clip|клип|plain/i.test(name) ? "plain" : "rich";
}

export function mockProbe(name: string, seedBase = 7): MediaInfo {
  const plain = fileKind(name) === "plain";
  const hash = [...name].reduce((a, c) => (a * 31 + c.charCodeAt(0)) >>> 0, seedBase);
  const duration = plain ? 48 + (hash % 30) : 1361;
  const audio: AudioStreamInfo[] = plain
    ? [{ index: 1, codec: "aac", channels: 2, channel_layout: "stereo", sample_rate: 48000, language: null, title: null, default: true }]
    : [
      { index: 1, codec: "eac3", channels: 6, channel_layout: "5.1", sample_rate: 48000, language: "eng", title: "English 5.1", default: true },
      { index: 2, codec: "aac", channels: 2, channel_layout: "stereo", sample_rate: 48000, language: "jpn", title: "Japanese", default: false },
      { index: 3, codec: "ac3", channels: 2, channel_layout: "stereo", sample_rate: 48000, language: "rus", title: "Русский дубляж", default: false },
    ];
  return {
    path: name, duration, container: plain ? "mov,mp4" : "matroska,webm", width: 1920, height: 1080, fps: 23.976, video_codec: "h264",
    audio_streams: audio,
    subtitle_streams: plain ? [] : [
      { index: 4, codec: "ass", language: "eng", title: "English (имена персонажей)", is_text: true, default: true, forced: false },
      { index: 5, codec: "subrip", language: "rus", title: null, is_text: true, default: false, forced: false },
      { index: 6, codec: "hdmv_pgs_subtitle", language: "jpn", title: null, is_text: false, default: false, forced: false },
    ],
    chapters: plain ? [] : ["Утро", "Письмо", "Парк", "Гараж", "Маяк", "Холодильник", "Дождь", "Финал"].map((title, i) => ({
      start: Math.round((i * duration) / 8), end: Math.round(((i + 1) * duration) / 8), title,
    })),
    size_bytes: plain ? 42_000_000 : 1_340_000_000,
  };
}

function overlapLines(an: Analysis, start: number, end: number): SceneLine[] {
  return an.lines
    .filter((l) => l.end > start && l.start < end)
    .map((l) => ({
      line_id: l.id, speaker_id: l.speaker_id ?? null,
      start_rel: Math.round(Math.max(0, l.start - start) * 100) / 100,
      end_rel: Math.round(Math.min(end, l.end) * 100 - start * 100) / 100,
      original: l.text, text: l.text,
    }));
}

type Listener = (j: Job) => void;

/** Текст движка (400) на склейку реплик разных персонажей. */
export const LINES_OTHER_SPEAKER_TEXT = "Это реплики разных персонажей — сначала смени персонажа у одной из них.";
const mergedOf = (an: Analysis): Record<string, string> => ((an as AnalysisX).merged_lines ??= {});

export class MockEngine implements Engine {
  readonly kind = "mock" as const;
  readonly base = "mock://engine";
  private speed: number;
  private ml: boolean;
  private projects = new Map<string, Project>();
  private jobs = new Map<string, Job>();
  private listeners = new Set<Listener>();
  private timers = new Set<ReturnType<typeof setInterval>>();
  private takeUrls = new Map<string, string>();
  private rendered = new Set<string>();
  private merges = new Map<string, Project>();
  private keys = new Map<string, KeyResult>();
  private seq = 1;
  private settings: AppSettings = {
    llm: {
      providers: [{ id: "claude_cli", kind: "claude_cli", title: "Claude (через Claude Code)", base_url: null, model: "sonnet", enabled: true, key_set: true, key_status: "ok" }],
      routing: { translate: "claude_cli", fit_length: "claude_cli", name_speakers: "claude_cli", director: "claude_cli" },
    },
    latency_ms: 120, target_lang: "ru", input_device: null, processing_device: "auto", whisper_model: "large-v3-turbo", separation_quality: "best", asr_engine: "auto",
  };

  constructor(opts: MockOptions = {}) {
    this.speed = opts.speed ?? 1;
    this.ml = opts.ml ?? true;
    if (opts.seed !== false) this.seed();
  }

  dispose() {
    this.timers.forEach((t) => clearInterval(t));
    this.timers.clear();
    this.listeners.clear();
  }

  // ───────────── начальные данные ─────────────
  private seed() {
    const media = mockProbe("Барсик и Мила — Серия 7.mkv");
    const an = buildAnalysis(1361, planFor("series"), 11);
    const p1: Project = {
      id: "p-fridge", title: "Барсик и Мила — Серия 7. Ограбление холодильника", source_path: "~/Movies/Барсик и Мила S01E07.mkv",
      media: { ...media, path: "~/Movies/Барсик и Мила S01E07.mkv", duration: 1361 }, analysis: an, analysis_state: "done", scenes: [],
      created: iso(new Date(Date.now() - 3 * 864e5)), updated: iso(new Date(Date.now() - 3600e3)),
    };
    this.projects.set(p1.id, p1);
    const sc = this.buildScene(p1, "sc001", "Холодильник открыт", FRIDGE_AT - 0.2, FRIDGE_AT + 21.7);
    sc.stems_ready = true;
    sc.prepare_state = "ready";
    sc.dubbed_speakers = ["s1", "s2"];
    sc.lines.forEach((l) => { l.text = RU_BY_EN[l.original] ?? l.text; });
    sc.mix.autopilot = true;
    p1.scenes.push(sc);
    // два дубля на первую реплику Милы и один на Барсика
    this.seedTake(p1, sc, sc.lines[0], 41, 2.0, { offset_ms: -80, stretch: 1, peak: -6.2, lufs: -21.4, selected: false, review: { text_match: 0.55, complete: false } });
    this.seedTake(p1, sc, sc.lines[0], 42, 2.0, { offset_ms: -35, stretch: 1.02, peak: -5.1, lufs: -20.3, selected: true, review: { text_match: 1, complete: true } });
    this.seedTake(p1, sc, sc.lines[1], 43, 2.0, { offset_ms: 210, stretch: 0.93, peak: -3.9, lufs: -19.5, selected: true });

    const m2 = mockProbe("Погоня в парке — клип.mp4");
    const p2: Project = {
      id: "p-park", title: "Погоня в парке (клип)", source_path: "~/Movies/Погоня в парке — клип.mp4",
      media: { ...m2, duration: 252, path: "~/Movies/Погоня в парке — клип.mp4" },
      analysis: buildAnalysis(252, planFor("short"), 5), analysis_state: "done", scenes: [], created: iso(new Date(Date.now() - 864e5)), updated: iso(new Date(Date.now() - 20 * 3600e3)),
    };
    const sc2 = this.buildScene(p2, "sc001", "Весь клип", 0, 252);
    sc2.stems_ready = true;
    sc2.prepare_state = "ready";
    p2.scenes.push(sc2);
    this.projects.set(p2.id, p2);

    // третий проект разбирается прямо сейчас
    const m3 = mockProbe("Барсик и Мила — Серия 8.mkv");
    const p3: Project = {
      id: "p-next", title: "Барсик и Мила — Серия 8. Новый сосед", source_path: "~/Movies/Барсик и Мила S01E08.mkv",
      media: { ...m3, path: "~/Movies/Барсик и Мила S01E08.mkv", duration: 1385 }, analysis: null, analysis_state: "running", scenes: [], created: iso(), updated: iso(),
    };
    this.projects.set(p3.id, p3);
    this.startJob("analyze", p3.id, null, ANALYZE_STEPS, () => {
      const q = this.projects.get(p3.id);
      if (q) { q.analysis = buildAnalysis(1385, planFor("next"), 21); q.analysis_state = "done"; q.updated = iso(); }
    });
  }

  private seedTake(p: Project, sc: Scene, line: SceneLine, seed: number, leadIn: number, v: { offset_ms: number; stretch: number; peak: number; lufs: number; selected: boolean; review?: Partial<Take> }) {
    const dur = line.end_rel - line.start_rel;
    const x = synthVoice(dur, leadIn, seed);
    const id = `t${String(sc.takes.length + 1).padStart(3, "0")}`;
    const url = typeof URL.createObjectURL === "function" ? URL.createObjectURL(wavBlob(x, 48000)) : null;
    if (url) this.takeUrls.set(`${p.id}/${sc.id}/${id}`, url);
    sc.takes.push({
      id, line_id: line.line_id, speaker_id: line.speaker_id, file: `takes/${id}.wav`, created: iso(), duration: Math.round((x.length / 48000) * 100) / 100,
      preroll: 1, start_rel: Math.max(0, line.start_rel - leadIn), offset_ms: v.offset_ms, stretch: v.stretch, fit_confidence: 0.82,
      peak_db: v.peak, loudness_lufs: v.lufs, recognized_text: null, selected: v.selected,
      source: "mic", empty: false, chosen_by_user: false, tail_trimmed: false, ...(v.review ?? {}),
    });
  }

  private buildScene(p: Project, id: string, title: string, start: number, end: number): Scene {
    const an = p.analysis!;
    start = Math.max(0, Math.round(start * 100) / 100);
    end = Math.min(p.media.duration, Math.round(end * 100) / 100);
    const lines = overlapLines(an, start, end);
    return {
      id, title, start, end, text_mode: "as_is", dubbed_speakers: [], lines, takes: [], mix: defaultMix() as MixSettings,
      stems_ready: false, prepare_state: "running", created: iso(),
    };
  }

  // ───────────── задачи ─────────────
  private emit(j: Job) {
    const c = clone(j);
    this.listeners.forEach((l) => l(c));
  }

  private startJob(kind: string, pid: string, scid: string | null, steps: [string, string][], onDone?: () => void, result?: Record<string, unknown>,
                   fail?: { step: string; error: string; code?: string }): Job {
    const job: Job = {
      id: `j${String(this.seq++).padStart(4, "0")}`, kind, project_id: pid, scene_id: scid, state: "running", error: null, result: result ?? null,
      steps: steps.map(([key, title]) => ({ key, title, progress: 0, state: "queued" as const })), created: iso(),
      progress: 0, eta_s: null, error_debug: null, error_code: null,
    };
    const started = Date.now();
    job.steps[0].state = "running";
    this.jobs.set(job.id, job);
    const rate = steps.map(() => 0.9 + Math.random() * 1.4);
    let cur = 0;
    const tick = setInterval(() => {
      if (job.state !== "running") { clearInterval(tick); this.timers.delete(tick); return; }
      const st: JobStep = job.steps[cur];
      st.progress = Math.min(1, st.progress + (0.1 * this.speed) / rate[cur]);
      if (fail && st.key === fail.step && st.progress >= 0.3) {
        st.state = "failed";
        job.state = "failed";
        job.error = fail.error;
        job.error_code = fail.code ?? null;
        clearInterval(tick); this.timers.delete(tick);
        this.emit(job);
        return;
      }
      job.progress = job.steps.reduce((a, x) => a + (x.state === "done" ? 1 : x.progress), 0) / job.steps.length;
      const spent = (Date.now() - started) / 1000;
      job.eta_s = job.progress > 0.08 && spent > 2 ? (spent / job.progress) * (1 - job.progress) : null;
      if (st.progress >= 1) {
        st.state = "done";
        cur++;
        if (cur >= job.steps.length) {
          job.state = "done";
          job.progress = 1;
          job.eta_s = null;
          try { onDone?.(); } catch (e) { job.state = "failed"; job.error = (e as Error).message; }
          clearInterval(tick); this.timers.delete(tick);
        } else job.steps[cur].state = "running";
      }
      this.emit(job);
    }, 100);
    this.timers.add(tick);
    this.emit(job);
    return clone(job);
  }

  async listJobs() { return [...this.jobs.values()].map(clone); }
  async getJob(id: string) {
    const j = this.jobs.get(id);
    if (!j) throw new EngineError("Задача не найдена.");
    return clone(j);
  }
  async cancelJob(id: string) {
    const j = this.jobs.get(id);
    if (j && j.state === "running") { j.state = "cancelled"; j.steps.forEach((s) => { if (s.state !== "done") s.state = "cancelled"; }); this.emit(j); }
  }
  subscribeJobs(cb: Listener) {
    this.listeners.add(cb);
    return () => { this.listeners.delete(cb); };
  }

  // ───────────── проекты ─────────────
  async health(): Promise<Health> {
    const ml = this.ml;
    return { ok: true, version: "demo", device: "demo", capabilities: { ml, asr: ml, separation: ml, diarization: ml, gigaam: ml, gigaam_downloaded: false, gigaam_download_mb: 230 } };
  }
  /** Тесты: «нейросети поставили» / «убрали». */
  setMl(ml: boolean) { this.ml = ml; }
  private mlFail() { return this.ml ? undefined : { step: "transcribe", error: ML_MISSING_TEXT, code: "ml_missing" }; }
  /** Как движок узнаёт язык и выбирает распознаватель (analysis/speech.py): выбор человека — закон; иначе «слушаем речь»
   *  (в демо: «дубляж»/«мегамозг» в имени файла — русская речь), метка дорожки — только подсказка. */
  private speechInfo(name: string, trackCode: string | null, chosen: string | null): Partial<Analysis> {
    const iso: Record<string, string> = { eng: "en", rus: "ru", jpn: "ja" };
    const track = trackCode ? iso[trackCode] ?? trackCode : null;
    const heard = /дубляж|мегамозг|megamind/i.test(name) ? "ru" : track ?? "en";
    const language = chosen || heard;
    const source = chosen ? "user" : "speech";
    const names: Record<string, string> = { ru: "русский", en: "английский", ja: "японский" };
    const note = source === "speech" && track && track !== language
      ? `В файле написано «${names[track] ?? track}», а слышно ${names[language] ?? language} — взял ${names[language] ?? language}. Не так? Выбери язык и разбери заново.` : null;
    const eng = this.settings.asr_engine ?? "auto";
    const asr = eng !== "whisper" && language === "ru" ? "gigaam" : "whisper";
    const notes = eng === "gigaam" && language !== "ru" ? ["GigaAM знает только русский — взял Whisper."] : [];
    return { language, language_source: source, track_language: track, language_note: note, asr_engine: asr, asr_notes: notes };
  }

  async reanalyze(pid: string, opts?: ReanalyzeOptions): Promise<Job> {
    const p = this.get(pid);
    if ([...this.jobs.values()].some((j) => j.kind === "analyze" && j.project_id === pid && j.state === "running")) throw new EngineError("Разбор этого проекта уже идёт.");
    const media = p.media;
    const plain = fileKind((p.source_path.split(/[\\/]/).pop() ?? "")) === "plain";
    const audioIdx = opts?.audio_stream_index ?? p.analysis?.audio_stream_index ?? media.audio_streams[0].index;
    return this.startJob("analyze", pid, null, ANALYZE_STEPS, () => {
      p.analysis = buildAnalysis(media.duration, plain ? planFor("short").map((x) => ({ ...x, at: x.at % Math.max(20, media.duration - 12) })) : planFor("series"), 3 + this.seq, audioIdx);
      Object.assign(p.analysis, this.speechInfo(p.source_path, media.audio_streams.find((a) => a.index === audioIdx)?.language ?? null, opts?.language ?? null));
      this.touch(p);
    }, undefined, this.mlFail());
  }
  async probe(src: { path?: string; url?: string; name?: string }): Promise<MediaInfo | UrlProbe | null> {
    await sleep(350 / this.speed);
    if (src.url) return { known: false, message: "Название и длину узнаем после скачивания.", subtitle_langs: [] };
    return mockProbe(src.name ?? src.path ?? "video.mkv");
  }

  async listProjects(): Promise<ProjectSummary[]> {
    return [...this.projects.values()]
      .map((p): ProjectSummaryX => ({
        id: p.id, title: p.title, duration: p.media.duration, scene_count: p.scenes.length, has_analysis: !!p.analysis,
        analysis_state: p.analysis_state ?? (p.analysis ? "done" : "none"), updated: p.updated ?? iso(), thumbnail: null, created: p.created ?? null,
      }))
      .sort((a, b) => b.updated.localeCompare(a.updated));
  }

  private get(pid: string): Project {
    const p = this.projects.get(pid);
    if (!p) throw new EngineError("Проект не найден.");
    return p;
  }
  private scene(pid: string, scid: string): [Project, Scene] {
    const p = this.get(pid);
    const s = p.scenes.find((x) => x.id === scid);
    if (!s) throw new EngineError("Сцена не найдена.");
    return [p, s];
  }
  private touch(p: Project) { p.updated = iso(); }

  async createProject(r: CreateProjectOptions): Promise<ProjectWithJob> {
    const src = r.path ?? r.url;
    if (!src) throw new EngineError("Укажи файл или ссылку.");
    const name = src.split(/[\\/]/).pop() ?? src;
    const media = mockProbe(name);
    media.path = src;
    const id = `p-${this.seq++}-${Math.random().toString(36).slice(2, 6)}`;
    const title = r.title?.trim() || name.replace(/\.[A-Za-z0-9]{2,4}$/, "") || "Новый проект";
    const p: Project = { id, title, source_path: src, media, analysis: null, analysis_state: "running", scenes: [], created: iso(), updated: iso() };
    this.projects.set(id, p);
    const plain = fileKind(name) === "plain";
    const audioIdx = r.audio_stream_index ?? media.audio_streams[0].index;
    const job = this.startJob("analyze", id, null, ANALYZE_STEPS, () => {
      p.analysis = buildAnalysis(media.duration, plain ? planFor("short").map((x) => ({ ...x, at: x.at % Math.max(20, media.duration - 12) })) : planFor("series"), 3 + this.seq, audioIdx);
      Object.assign(p.analysis, this.speechInfo(name, media.audio_streams.find((a) => a.index === audioIdx)?.language ?? null, r.language ?? null));
      this.touch(p);
    }, undefined, this.mlFail());
    return { project: clone(p), job };
  }
  async getProject(id: string) { return clone(this.get(id)); }
  async deleteProject(id: string) {
    this.get(id);
    this.projects.delete(id);
  }

  async renameProject(id: string, title: string) {
    const p = this.get(id);
    const t = title.trim();
    if (!t) throw new EngineError("Название не может быть пустым.");
    p.title = t;
    this.touch(p);
    return clone(p);
  }

  async renameSpeaker(pid: string, sid: string, name: string) {
    const p = this.get(pid);
    const sp = p.analysis?.speakers.find((s) => s.id === sid);
    if (!sp) throw new EngineError("Такого говорящего нет.");
    const n = name.trim();
    sp.name = n || null;
    sp.label = n || SPEAKER_DEFS.find((d) => d.id === sid)?.label || sp.label;
    this.touch(p);
    return clone(p);
  }
  async mergeSpeakers(pid: string, a: string, b: string): Promise<MergeResult> {
    const p = this.get(pid);
    const an = p.analysis;
    if (!an) throw new EngineError("Проект ещё не разобран.");
    if (a === b) throw new EngineError("Выбери двух разных говорящих.");
    const mergeId = `m${this.seq++}`;
    this.merges.set(`${pid}/${mergeId}`, clone(p));
    an.lines.forEach((l) => { if (l.speaker_id === b) l.speaker_id = a; });
    an.speakers = an.speakers.filter((s) => s.id !== b);
    const keep = an.speakers.find((s) => s.id === a);
    if (keep) {
      const mine = an.lines.filter((l) => l.speaker_id === a);
      keep.line_count = mine.length;
      keep.total_speech = Math.round(mine.reduce((x, l) => x + l.end - l.start, 0) * 10) / 10;
    }
    an.merge_suggestions = an.merge_suggestions.filter((m) => !m.slice(0, 2).includes(b));
    p.scenes.forEach((sc) => {
      sc.lines.forEach((l) => { if (l.speaker_id === b) l.speaker_id = a; });
      sc.takes.forEach((t) => { if (t.speaker_id === b) t.speaker_id = a; });
      sc.dubbed_speakers = [...new Set(sc.dubbed_speakers.map((s) => (s === b ? a : s)))];
    });
    this.touch(p);
    return { project: clone(p), merge_id: mergeId };
  }
  async undoMerge(pid: string, mergeId: string) {
    const before = this.merges.get(`${pid}/${mergeId}`);
    if (!before) throw new EngineError("Это слияние уже нельзя отменить.");
    this.merges.delete(`${pid}/${mergeId}`);
    const p = this.get(pid);
    p.analysis = before.analysis;
    p.scenes = before.scenes;
    this.touch(p);
    return clone(p);
  }

  private recount(p: Project) {
    const an = p.analysis;
    if (!an) return;
    for (const sp of an.speakers) {
      const mine = an.lines.filter((l) => l.speaker_id === sp.id);
      sp.line_count = mine.length;
      sp.total_speech = Math.round(mine.reduce((x, l) => x + l.end - l.start, 0) * 10) / 10;
    }
  }
  async setLineSpeaker(pid: string, lineId: string, target: { speaker_id?: string; new_speaker_name?: string }) {
    const p = this.get(pid);
    const an = p.analysis;
    if (!an) throw new EngineError("Проект ещё не разобран.");
    const ln = an.lines.find((l) => l.id === lineId);
    if (!ln) throw new EngineError("Такой реплики нет.");
    let sid = target.speaker_id ?? null;
    const name = target.new_speaker_name?.trim();
    if (name) {
      const same = an.speakers.find((s) => (s.name ?? s.label).toLowerCase() === name.toLowerCase());
      if (same) sid = same.id;
      else {
        let n = 1;
        while (an.speakers.some((s) => s.id === `s${n}`)) n++;
        const colors = ["#4fb8a8", "#e57a5f", "#6aa6e8", "#c58ad6", "#a8c95a", "#e8c25a", "#e87fa6"];
        const color = colors.find((c) => !an.speakers.some((s) => s.color === c)) ?? colors[0];
        an.speakers.push({ id: `s${n}`, label: name, name, color, total_speech: 0, line_count: 0 });
        sid = `s${n}`;
      }
    }
    if (!sid || !an.speakers.some((s) => s.id === sid)) throw new EngineError("Такого персонажа нет.");
    const old = ln.speaker_id;
    ln.speaker_id = sid;
    for (const sc of p.scenes) {
      if (!sc.lines.some((l) => l.line_id === lineId)) continue;
      sc.lines.forEach((l) => { if (l.line_id === lineId) l.speaker_id = sid; });
      sc.takes.forEach((t) => { if (t.line_id === lineId) t.speaker_id = sid; });
      if ((!old || sc.dubbed_speakers.includes(old)) && !sc.dubbed_speakers.includes(sid)) sc.dubbed_speakers.push(sid);
    }
    this.recount(p);
    this.touch(p);
    return clone(p);
  }
  async mergeLines(pid: string, lineIds: string[]): Promise<LinesMergeResult> {
    const p = this.get(pid);
    const an = p.analysis;
    if (!an) throw new EngineError("Проект ещё не разобран.");
    const order = [...an.lines].sort((a, b) => a.start - b.start);
    const idx = [...new Set(lineIds)].map((id) => order.findIndex((l) => l.id === id)).sort((a, b) => a - b);
    if (idx.length < 2 || idx.includes(-1)) throw new EngineError("Чтобы склеить, нужны хотя бы две реплики.");
    if (idx[idx.length - 1] - idx[0] !== idx.length - 1) throw new EngineError("Склеить можно только реплики, которые идут подряд.");
    const parts = order.slice(idx[0], idx[idx.length - 1] + 1);
    // как движок: реплики разных персонажей не склеиваем — иначе дубль второго молча пропал бы из ролика
    if (new Set(parts.map((l) => l.speaker_id ?? "")).size > 1) throw new EngineError(LINES_OTHER_SPEAKER_TEXT);
    const ids = parts.map((l) => l.id);
    const mergeId = `j${this.seq++}`;
    this.merges.set(`${pid}/${mergeId}`, clone(p));
    const keep = parts[0];
    const text = parts.map((l) => l.text.trim()).join(" ");
    an.lines = an.lines.filter((l) => !ids.slice(1).includes(l.id));
    Object.assign(keep, { end: parts[parts.length - 1].end, text, words: parts.flatMap((l) => l.words) });
    let moved = 0;
    for (const sc of p.scenes) {
      const mine = sc.lines.filter((l) => ids.includes(l.line_id));
      if (!mine.length) continue;
      const first = mine[0];
      const untouched = mine.every((l) => l.text === l.original);
      Object.assign(first, { end_rel: mine[mine.length - 1].end_rel, original: text, speaker_id: keep.speaker_id,
        text: untouched ? text : mine.map((l) => l.text.trim()).join(" ") });
      sc.lines = sc.lines.filter((l) => l === first || !ids.includes(l.line_id));
      for (const t of sc.takes) if (ids.includes(t.line_id ?? "")) { if (t.line_id !== keep.id) moved++; t.line_id = keep.id; t.speaker_id = keep.speaker_id ?? null; }
      const group = sc.takes.filter((t) => t.line_id === keep.id);
      const star = group.find((t) => t.selected && t.chosen_by_user) ?? group.find((t) => t.selected);
      group.forEach((t) => { t.selected = t === star; });
    }
    // как движок (Analysis.merged_lines): склеенная реплика → последняя склейка, по которой её можно разделить
    const ml = mergedOf(an);
    for (const id of ids) delete ml[id];
    ml[keep.id] = mergeId;
    this.recount(p);
    this.touch(p);
    return { project: clone(p), merge_id: mergeId, moved_takes: moved };
  }
  /** «Разделить обратно»: отмена последней склейки, в которую вошла реплика (POST …/lines/{id}/split). */
  async splitLine(pid: string, lineId: string) {
    const p = this.get(pid);
    const mergeId = p.analysis ? mergedOf(p.analysis)[lineId] : undefined;
    if (!mergeId || !this.merges.has(`${pid}/${mergeId}`)) throw new EngineError("Эта реплика не склеивалась — разделять нечего.");
    return this.undoMergeLines(pid, mergeId);
  }
  async undoMergeLines(pid: string, mergeId: string) {
    const before = this.merges.get(`${pid}/${mergeId}`);
    if (!before) throw new EngineError("Такой склейки нет или её уже отменили.");
    this.merges.delete(`${pid}/${mergeId}`);
    const p = this.get(pid);
    p.analysis = before.analysis;
    p.scenes = before.scenes;
    this.touch(p);
    return clone(p);
  }

  async snap(pid: string, r: SnapRequest): Promise<SnapResult> {
    const p = this.get(pid);
    const an = p.analysis;
    if (!an) throw new EngineError("Проект ещё не разобран.");
    const [start, end] = selectByMode(r.mode, r.a, r.a, r.b ?? r.a, an.lines, an.shot_cuts, p.media.duration);
    return { start, end };
  }

  // ───────────── сцены ─────────────
  async createScene(pid: string, r: SceneCreate): Promise<SceneWithJob> {
    const p = this.get(pid);
    if (!p.analysis) throw new EngineError("Проект ещё не разобран.");
    if (!(r.end > r.start) || r.start < 0 || r.end > p.media.duration + 0.01) throw new EngineError("Некорректные границы сцены.");
    const n = p.scenes.length + 1;
    let id = `sc${String(n).padStart(3, "0")}`;
    while (p.scenes.some((s) => s.id === id)) id += "x";
    const sc = this.buildScene(p, id, r.title.trim() || `Сцена ${n}`, r.start, r.end);
    p.scenes.push(sc);
    this.touch(p);
    const job = this.startJob("prepare_scene", pid, id, PREPARE_STEPS, () => { sc.stems_ready = true; sc.prepare_state = "ready"; this.touch(p); });
    return { scene: clone(sc), job };
  }
  async prepareScene(pid: string, scid: string): Promise<Job> {
    const [p, sc] = this.scene(pid, scid);
    sc.stems_ready = false;
    sc.prepare_state = "running";
    return this.startJob("prepare_scene", pid, scid, PREPARE_STEPS, () => { sc.stems_ready = true; sc.prepare_state = "ready"; this.touch(p); });
  }
  async getScene(pid: string, scid: string) { return clone(this.scene(pid, scid)[1]); }
  async patchScene(pid: string, scid: string, patch: ScenePatch) {
    const [p, sc] = this.scene(pid, scid);
    if (patch.title != null) sc.title = patch.title;
    if (patch.dubbed_speakers != null) sc.dubbed_speakers = patch.dubbed_speakers;
    if (patch.lines) {
      for (const lp of patch.lines) {
        const l = sc.lines.find((x) => x.line_id === lp.line_id);
        if (!l) throw new EngineError(`В сцене нет реплики ${lp.line_id}.`);
        l.text = lp.text;
        l.translation_source = "user";
      }
    }
    if (patch.text_mode != null && patch.text_mode !== sc.text_mode) {
      sc.text_mode = patch.text_mode;
      if (patch.text_mode === "as_is" && !patch.lines) {
        // «Как есть»: вернули оригинал или его перевод
        sc.lines.forEach((l) => { l.text = this.translated.get(`${pid}/${scid}/${l.line_id}`) ?? l.original; });
      }
    }
    if (patch.mix) {
      if (JSON.stringify(patch.mix) !== JSON.stringify(sc.mix)) sc.mix_check = null; // как движок: проверка звука устарела
      sc.mix = clone(patch.mix);
      if (sc.mix.autopilot) this.automix(sc);
    }
    this.touch(p);
    return clone(sc);
  }
  private translated = new Map<string, string>();

  /** Имитация audio.mix.automix: громкость подбирает сам, а комнату, эффект, «выкл.» и «только она» оставляет как выбрал человек. */
  private automix(sc: Scene) {
    sc.mix.background = { ...sc.mix.background, gain_db: -6 };
    for (const sid of sc.dubbed_speakers) {
      const t = sc.takes.find((x) => x.speaker_id === sid && x.selected);
      const tr = sc.mix.tracks[sid] ?? defaultTrack();
      const loud = t?.loudness_lufs ?? -20;
      sc.mix.tracks[sid] = { ...tr, gain_db: Math.round(clamp(-20 - loud, -9, 9) * 2) / 2 };
    }
  }
  async deleteScene(pid: string, scid: string) {
    const [p, sc] = this.scene(pid, scid);
    p.scenes = p.scenes.filter((s) => s !== sc);
    this.touch(p);
  }

  // ───────────── дубли ─────────────
  async uploadTake(pid: string, scid: string, wav: Blob, meta: TakeUploadMeta): Promise<Take> {
    const [p, sc] = this.scene(pid, scid);
    const awaitedBuf = await blobToArrayBuffer(wav);
    // настоящий движок понимает любой звук (ffmpeg); демо — только WAV, остальное заменяем короткой «речью»
    const { samples, sampleRate } = (() => {
      try { return decodeWav(awaitedBuf); } catch {
        const sr = 48000, n = sr * 2, out = new Float32Array(n);
        for (let i = 0; i < n; i++) out[i] = 0.25 * Math.sin((2 * Math.PI * 180 * i) / sr) * (0.6 + 0.4 * Math.sin((2 * Math.PI * 3 * i) / sr));
        return { samples: out, sampleRate: sr };
      }
    })();
    const line = sc.lines.find((l) => l.line_id === meta.line_id) ?? null;
    if (peakDb(samples) < -55) throw new EngineError("Запись почти пустая — микрофон ничего не услышал. Проверь, что выбран нужный микрофон, и запиши ещё раз.");
    const bounds = speechBounds(samples, sampleRate);
    const fileT0 = meta.start_rel - meta.preroll; // сцена-время нулевого сэмпла файла
    let offset = 0;
    let stretch = 1;
    let conf = 0;
    if (line && bounds) {
      const sStart = fileT0 + bounds[0];
      offset = clamp(Math.round((line.start_rel - sStart) * 1000), -1500, 1500);
      const lineDur = line.end_rel - line.start_rel;
      const speech = Math.max(0.2, bounds[1] - bounds[0]);
      stretch = clamp(Math.round((lineDur / speech) * 100) / 100, 0.85, 1.15);
      if (Math.abs(stretch - 1) < 0.03) stretch = 1;
      conf = Math.round(clamp(0.55 + Math.min(0.4, (speech / lineDur) * 0.3), 0, 1) * 100) / 100;
    }
    const id = `t${String(sc.takes.length + 1).padStart(3, "0")}`;
    const speechPart = bounds ? samples.slice(Math.floor(bounds[0] * sampleRate), Math.ceil(bounds[1] * sampleRate)) : samples;
    const take: Take = {
      id, line_id: line?.line_id ?? null, speaker_id: line?.speaker_id ?? null, file: `takes/${id}.wav`, created: iso(),
      duration: Math.round((samples.length / sampleRate) * 100) / 100, preroll: meta.preroll, start_rel: meta.start_rel,
      offset_ms: offset, stretch, fit_confidence: conf, peak_db: Math.round(peakDb(samples) * 10) / 10,
      loudness_lufs: Math.round((rmsDb(speechPart) - 0.7) * 10) / 10, recognized_text: null, selected: true,
      source: meta.source ?? (typeof File !== "undefined" && wav instanceof File ? "file" : "mic"), empty: false, chosen_by_user: false, tail_trimmed: false,
    };
    sc.takes.forEach((t) => { if (t.line_id === take.line_id) t.selected = false; });
    sc.takes.push(take);
    if (typeof URL.createObjectURL === "function") this.takeUrls.set(`${pid}/${scid}/${id}`, URL.createObjectURL(wav));
    if (sc.mix.autopilot) this.automix(sc);
    this.touch(p);
    await sleep(250 / this.speed);
    return clone(take);
  }
  async patchTake(pid: string, scid: string, tid: string, patch: TakePatch) {
    const [p, sc] = this.scene(pid, scid);
    const t = sc.takes.find((x) => x.id === tid);
    if (!t) throw new EngineError("Дубль не найден.");
    if (patch.offset_ms != null) t.offset_ms = patch.offset_ms;
    if (patch.stretch != null) t.stretch = patch.stretch;
    if (patch.selected != null) {
      if (patch.selected) sc.takes.forEach((o) => { if (o.line_id === t.line_id && o.id !== t.id) o.selected = false; });
      t.selected = patch.selected;
      if (sc.mix.autopilot) this.automix(sc);
    }
    this.touch(p);
    return clone(t);
  }
  async untrimTakeTail(pid: string, scid: string, tid: string): Promise<Take> {
    const [p, sc] = this.scene(pid, scid);
    const t = sc.takes.find((x) => x.id === tid);
    if (!t) throw new EngineError("Дубль не найден.");
    t.tail_trimmed = false;
    this.touch(p);
    return clone(t);
  }
  async micCheck(wav: Blob, silenceS: number): Promise<MicCheckResult> {
    const { samples, sampleRate } = decodeWav(await blobToArrayBuffer(wav));
    const cut = Math.floor(silenceS * sampleRate);
    const noise = rmsDb(samples.slice(0, cut));
    const speech = samples.slice(cut);
    const peak = peakDb(speech);
    const found = peak > -50;
    const verdict = !found ? "silent" : peak > -3 ? "loud" : peak < -26 || rmsDb(speech) < -30 ? "quiet" : noise > -45 ? "noisy" : "good";
    const advice = { good: "Всё хорошо: голос слышно, в комнате тихо.", quiet: "Микрофон тихий: прибавь ему громкость в настройках звука или подойди ближе к микрофону — на ширину ладони.", loud: "Громковато — отодвинься от микрофона.", noisy: "В комнате шумно — закрой окно и выключи вентилятор.", silent: "Не слышу твой голос — проверь микрофон." }[verdict];
    return { noise_floor_db: noise, speech_db: rmsDb(speech), peak_db: peak, snr_db: rmsDb(speech) - noise, clipping: peak > -0.3, speech_found: found, verdict, advice, saved: found } as MicCheckResult;
  }
  async micSignal(): Promise<ArrayBuffer> { return new ArrayBuffer(0); }
  async micLatency(): Promise<LatencyResult> { return { latency_ms: 120, confidence: 0.9, ok: true, message: "Задержка 120 мс", saved: true }; }
  async trimTakeTail(pid: string, scid: string, tid: string): Promise<Take> {
    const [p, sc] = this.scene(pid, scid);
    const t = sc.takes.find((x) => x.id === tid);
    if (!t) throw new EngineError("Дубль не найден.");
    t.tail_cut_s = null;
    t.tail_text = null;
    t.tail_trimmed = true;
    t.complete = true;
    this.touch(p);
    return clone(t);
  }
  async deleteTake(pid: string, scid: string, tid: string) {
    const [p, sc] = this.scene(pid, scid);
    sc.takes = sc.takes.filter((t) => t.id !== tid);
    this.takeUrls.delete(`${pid}/${scid}/${tid}`);
    this.touch(p);
  }

  async preview(pid: string, scid: string): Promise<Job> {
    this.scene(pid, scid);
    return this.startJob("preview", pid, scid, RENDER_STEPS, undefined, { path: "preview.mp4", url: "mock" });
  }
  async render(pid: string, r: RenderProjectRequest): Promise<Job> {
    this.get(pid);
    if (!r.scene_ids.length) throw new EngineError("Выбери хотя бы одну сцену.");
    r.scene_ids.forEach((s) => this.scene(pid, s));
    const steps: [string, string][] = r.scene_ids.length > 1 ? [...RENDER_STEPS, ["concat", "Склейка сцен в один ролик"]] : RENDER_STEPS;
    // как настоящий движок: файл с таким именем уже есть и overwrite не включён → « (2)», « (3)»
    let path = r.settings.output_path;
    if (!r.settings.overwrite && this.rendered.has(path)) {
      const m = path.match(/^(.*?)(\.[^./\\]+)?$/)!;
      let n = 2;
      while (this.rendered.has(`${m[1]} (${n})${m[2] ?? ""}`)) n++;
      path = `${m[1]} (${n})${m[2] ?? ""}`;
    }
    this.rendered.add(path);
    return this.startJob("render_project", pid, null, steps, undefined, { path, url: null });
  }

  // ───────────── ИИ ─────────────
  private async think() { await sleep((500 + Math.random() * 500) / this.speed); }

  async translate(r: LLMRequest): Promise<TranslateResult> {
    await this.think();
    const p = this.get(r.project_id);
    const [, sc] = this.scene(r.project_id, r.scene_id ?? "");
    const translations: { line_id: string; text: string }[] = [];
    for (const l of sc.lines) {
      if (r.line_ids && !r.line_ids.includes(l.line_id)) continue;
      const ru = RU_BY_EN[l.original] ?? l.original;
      translations.push({ line_id: l.line_id, text: ru });
      this.translated.set(`${r.project_id}/${r.scene_id}/${l.line_id}`, ru);
      if (sc.text_mode === "as_is") l.text = ru;
      l.translation_source = "llm";
    }
    this.touch(p);
    return { translations, ai_fallback: false };
  }
  async fitLength(r: LLMRequest): Promise<FitLengthResult> {
    await this.think();
    const [, sc] = this.scene(r.project_id, r.scene_id ?? "");
    const variants: { line_id: string; texts: string[] }[] = [];
    for (const l of sc.lines) {
      if (r.line_ids && !r.line_ids.includes(l.line_id)) continue;
      variants.push({ line_id: l.line_id, texts: shortenVariants(l.text, l.end_rel - l.start_rel, r.n ?? 3) });
    }
    return { variants, ai_fallback: false };
  }
  async nameSpeakers(r: LLMRequest): Promise<NameSpeakersResult> {
    await this.think();
    const p = this.get(r.project_id);
    const names: Record<string, string> = {};
    const guesses: Record<string, string> = { s5: "Тётя Зина", s6: "Барсик (кричит)" };
    for (const s of p.analysis?.speakers ?? []) if (!s.name && guesses[s.id]) names[s.id] = guesses[s.id];
    return { names, ai_fallback: false };
  }
  async directorTake(r: LLMRequest): Promise<DirectorTakeResult> {
    await this.think();
    const [, sc] = this.scene(r.project_id, r.scene_id ?? "");
    const t = sc.takes.find((x) => x.id === r.take_id);
    if (!t) throw new EngineError("Дубль не найден.");
    const line = sc.lines.find((l) => l.line_id === t.line_id);
    const dur = line ? line.end_rel - line.start_rel : null;
    // как движок: дубль из файла к губам не привязан — «попал в губы» про него не говорим
    return { report: takeReport(t, t.line_id ?? "", dur), feedback: takeRemark(t, dur, t.source === "file").text, ai_fallback: false };
  }
  async directorCommand(r: LLMRequest): Promise<DirectorCommandResult> {
    await this.think();
    const p = this.get(r.project_id);
    const [, sc] = this.scene(r.project_id, r.scene_id ?? "");
    return parseCommand(r.text ?? "", p.analysis?.speakers ?? [], sc.mix);
  }

  // ───────────── настройки ─────────────
  async getSettings() {
    const s = clone(this.settings);
    s.llm.providers.forEach((p) => {
      const k = this.keys.get(p.id);
      p.key_set = p.kind === "claude_cli" ? true : !!k;
      p.key_status = k?.key_status ?? "unknown";
      p.key_message = k?.key_message ?? null;
    });
    return s;
  }
  async putSettings(s: AppSettings) { this.settings = clone(s); return this.getSettings(); }
  async setKey(id: string, key: string): Promise<KeyResult> {
    if (!key.trim()) throw new EngineError("Ключ пустой.");
    // в демо ключ «проверяется» по виду: sk-test… считаем неподходящим
    const bad = /^sk-test|bad|wrong/i.test(key.trim());
    const r: KeyResult = bad
      ? { key_status: "invalid", key_message: "Модель не приняла этот ключ. Проверь, что скопировал его целиком, или возьми новый." }
      : { key_status: "ok", key_message: null };
    this.keys.set(id, r);
    return r;
  }
  async checkProvider(id: string): Promise<CheckResult> {
    await sleep(500 / this.speed);
    const cfg = this.settings.llm.providers.find((p) => p.id === id);
    if (!cfg) throw new EngineError("Такая модель не настроена.");
    const k = this.keys.get(id);
    if (k?.key_status === "invalid") return { ok: false, message: k.key_message ?? "Ключ не подходит.", key_status: "invalid", key_message: k.key_message };
    return { ok: true, message: "Ответила за 0,8 с", key_status: "ok", key_message: null };
  }

  // ───────────── медиа ─────────────
  sourceVideoUrl() { return null; }
  proxyVideo(): ProxySource | null { return null; }
  clipUrl() { return null; }
  previewUrl() { return null; }
  backgroundUrl() { return null; }
  takeUrl(pid: string, scid: string, t: Take) { return this.takeUrls.get(`${pid}/${scid}/${t.id}`) ?? null; }
  thumbnailUrl() { return null; }

  /** Для тестов и демо: всё ли готово */
  hasRunningJobs() { return [...this.jobs.values()].some((j) => j.state === "running"); }
}

// ───────────── «ИИ» для демо: сократить реплику ─────────────
const FILLERS = /\b(очень|просто|же|вот|ну|всё|уже|даже|тут|там|именно|как бы|прямо|совсем|конечно|кстати)\b[,]?\s*/gi;

export function shortenVariants(text: string, duration: number, n = 3): string[] {
  const lang = guessLang(text);
  const fits = (s: string) => estimateDuration(s, lang) <= duration * 1.05;
  const out: string[] = [];
  const push = (s: string) => {
    s = s.replace(/\s+/g, " ").replace(/\s+([,.!?…])/g, "$1").trim();
    if (s && s !== text && !out.includes(s)) out.push(s);
  };
  const noFill = text.replace(FILLERS, "");
  push(noFill);
  // по первой фразе
  const parts = noFill.split(/(?<=[.!?…—])\s+|,\s+/);
  for (let i = parts.length; i >= 1; i--) {
    const s = parts.slice(0, i).join(", ").replace(/[,—]$/, "");
    if (fits(s)) { push(/[.!?…]$/.test(s) ? s : s + "."); break; }
  }
  // по словам, пока не влезет
  const words = noFill.split(" ");
  while (words.length > 2 && !fits(words.join(" "))) words.pop();
  push(words.join(" ").replace(/[,—-]$/, "") + (/[.!?…]$/.test(words.join(" ")) ? "" : "…"));
  // запасной: самое короткое
  if (out.length < n) push(words.slice(0, Math.max(1, Math.ceil(words.length * 0.7))).join(" "));
  return out.slice(0, n);
}

// удобный экспорт для applyPatches в тестах
export { applyPatches };
