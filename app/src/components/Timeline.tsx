import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Chapter, Line, Scene, Speaker } from "../api/types";
import { fmt, fmtLong } from "../lib/format";
import { selectByMode, snapSelection } from "../lib/snap";

export type PickMode = "dialogue" | "shot" | "free";
export type Zoom = "all" | "5m" | "1m";

interface Props {
  duration: number;
  lines: Line[];
  speakers: Speaker[];
  cuts: number[];
  chapters?: Chapter[];
  scenes?: Scene[];
  mode: PickMode;
  selection: [number, number] | null;
  onSelect: (range: [number, number]) => void;
  playhead: number;
  onSeek: (t: number) => void;
  onOpenScene?: (id: string) => void;
  zoom: Zoom;
  /** подсветить сцены, выбранные для склейки */
  activeSceneId?: string | null;
  /** подсказка по центру шкалы (пока ничего не выбрано) */
  hint?: string | null;
}

const LABEL_W = 136;
const ROW = 22;
const ZOOM_SPAN: Record<Zoom, number> = { all: Infinity, "5m": 300, "1m": 60 };

function niceStep(span: number): number {
  const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
  return steps.find((s) => span / s <= 14) ?? 600;
}

/**
 * Шкала всего видео. Дорожка на каждого говорящего, склейки — тонкие риски по всей высоте.
 * Три способа выделить: «Диалог» (клик → весь разговор), «Сцена» (клик → от склейки до склейки),
 * «Вручную» (протянуть). Края прилипают к репликам и склейкам (lib/snap.ts).
 */
export function Timeline({ duration, lines, speakers, cuts, chapters = [], scenes = [], mode, selection, onSelect, playhead, onSeek, onOpenScene, zoom, activeSceneId, hint }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const tracksRef = useRef<HTMLDivElement>(null);
  const [viewW, setViewW] = useState(900);
  const [drag, setDrag] = useState<[number, number] | null>(null);
  const [hover, setHover] = useState<[number, number] | null>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const measure = () => setViewW(el.clientWidth || 900);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const span = Math.min(duration || 1, ZOOM_SPAN[zoom]);
  const innerW = Math.max(viewW, Math.round((viewW * (duration || 1)) / span));
  const pps = innerW / (duration || 1);
  const x = useCallback((t: number) => t * pps, [pps]);

  // при смене масштаба держим в кадре плейхед
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (zoom === "all") el.scrollLeft = 0;
    else el.scrollLeft = Math.max(0, playhead * pps - viewW / 2);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [zoom]);

  // новое выделение — показываем
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || !selection || zoom === "all") return;
    const a = selection[0] * pps;
    const b = selection[1] * pps;
    if (a < el.scrollLeft || b > el.scrollLeft + viewW) el.scrollLeft = Math.max(0, a - 40);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selection?.[0], selection?.[1]]);

  const timeAt = useCallback((clientX: number): number => {
    const r = tracksRef.current?.getBoundingClientRect();
    const left = r?.left ?? 0;
    return Math.min(duration, Math.max(0, (clientX - left) / pps));
  }, [pps, duration]);

  const speakerRows = useMemo(() => {
    const ids = new Set(lines.map((l) => l.speaker_id ?? "?"));
    const rows = speakers.filter((s) => ids.has(s.id));
    if (ids.has("?")) rows.push({ id: "?", label: "Неизвестно", name: null, color: "#85827d", line_count: 0, total_speech: 0 });
    return rows;
  }, [lines, speakers]);

  const rowIndex = (sid: string | null | undefined) => speakerRows.findIndex((s) => s.id === (sid ?? "?"));

  /**
   * Мышь на шкале: просто клик по реплике выделяет разговор (или кадр — по режиму), а протянуть мышью — выделить свой кусок.
   * Так «Создать сцену» появляется сразу, в любом режиме.
   */
  const onDown = (e: React.MouseEvent) => {
    if (e.button !== 0) return;
    const t0 = timeAt(e.clientX);
    if ((e.target as HTMLElement).closest("[data-scene],[data-chapter]")) return;
    onSeek(t0);
    let moved = false;
    setDrag([t0, t0]);
    const move = (ev: MouseEvent) => {
      const t1 = timeAt(ev.clientX);
      if (Math.abs(t1 - t0) * pps > 4) moved = true;
      if (moved) setDrag([t0, t1]);
    };
    const up = (ev: MouseEvent) => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      const t1 = timeAt(ev.clientX);
      setDrag(null);
      if (moved && Math.abs(t1 - t0) > 0.3) {
        onSelect(snapSelection(Math.min(t0, t1), Math.max(t0, t1), lines, cuts));
      } else {
        // клик: «Вручную» тоже выделяет разговор вокруг реплики
        const r = selectByMode(mode === "free" ? "dialogue" : mode, t0, t0, t0, lines, cuts, duration);
        if (r[1] > r[0]) onSelect(r);
      }
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  };

  const onMove = (e: React.MouseEvent) => {
    if (drag) { if (hover) setHover(null); return; }
    const r = selectByMode(mode === "free" ? "dialogue" : mode, timeAt(e.clientX), 0, 0, lines, cuts, duration);
    setHover(r[1] > r[0] ? r : null);
  };

  const step = niceStep(span);
  const ticks: number[] = [];
  for (let t = 0; t <= duration; t += step) ticks.push(t);

  const live = drag ? ([Math.min(...drag), Math.max(...drag)] as [number, number]) : null;
  const box = (r: [number, number], cls: string, label?: string) => (
    <div className={cls} style={{ left: x(r[0]), width: Math.max(3, x(r[1]) - x(r[0])) }} data-testid={cls}>{label}</div>
  );

  const hasChapters = chapters.length > 0;
  const labels: { key: string; node: React.ReactNode }[] = [
    { key: "ruler", node: <div className="rl" /> },
    { key: "scenes", node: <div>Твои сцены</div> },
    ...(hasChapters ? [{ key: "chap", node: <div>Главы файла</div> }] : []),
    ...speakerRows.map((s) => ({
      key: s.id,
      node: <div title={s.name ?? s.label}><i className="dot" style={{ background: s.color }} /><span className="ell">{s.name ?? s.label}</span></div>,
    })),
  ];

  return (
    <div className="tl-outer">
      {hint && !selection && !live && <div className="tl-hint" data-testid="tl-hint">{hint}</div>}
    <div className="tl-body">
      <div className="tl-labels" style={{ width: LABEL_W }}>
        {labels.map((l) => <div key={l.key} className={l.key === "ruler" ? "rl" : ""}>{l.key === "ruler" ? null : l.node}</div>)}
      </div>
      <div className="tl-scroll" ref={scrollRef}>
        <div className={"tl-tracks mode-" + (mode === "shot" ? "shot" : "dialogue")} ref={tracksRef} style={{ width: innerW }} onMouseDown={onDown} onMouseMove={onMove} onMouseLeave={() => setHover(null)}
          role="application" aria-label="Шкала видео">
          <div className="ruler" aria-hidden="true">
            {ticks.map((t) => <span key={t} style={{ left: x(t) }}>{span > 600 || duration > 3600 ? fmtLong(t) : fmt(t).replace(/\.\d$/, "")}</span>)}
          </div>
          <div className="mlane" aria-label="Сцены">
            {scenes.map((sc) => (
              <button key={sc.id} data-scene={sc.id} className={"scn" + (sc.id === activeSceneId ? " on" : "")} title={`${sc.title} · ${fmt(sc.start)}–${fmt(sc.end)}`}
                style={{ left: x(sc.start), width: Math.max(16, x(sc.end) - x(sc.start)) }}
                onClick={(e) => { e.stopPropagation(); onOpenScene?.(sc.id); }}>
                <span className="ell">{sc.title}</span>
              </button>
            ))}
          </div>
          {hasChapters && (
            <div className="mlane" aria-label="Главы">
              {chapters.map((c, i) => (
                <button key={i} data-chapter={i} className="chap" title={`${c.title ?? "Глава " + (i + 1)} · ${fmt(c.start)}–${fmt(c.end)}`}
                  style={{ left: x(c.start), width: Math.max(14, x(c.end) - x(c.start) - 2) }}
                  onClick={(e) => { e.stopPropagation(); onSelect(snapSelection(c.start, c.end, lines, cuts)); }}>
                  <span className="ell">{c.title ?? i + 1}</span>
                </button>
              ))}
            </div>
          )}
          {speakerRows.map((s) => (
            <div key={s.id} className="lane" data-lane={s.id}>
              {lines.filter((l) => rowIndex(l.speaker_id) === speakerRows.indexOf(s)).map((l) => (
                <i key={l.id} title={l.text} style={{ left: x(l.start), width: Math.max(2, x(l.end) - x(l.start)), background: s.color }} />
              ))}
            </div>
          ))}
          {cuts.map((c) => <b key={c} className="cut" style={{ left: x(c) }} aria-hidden="true" />)}
          {hover && !selection && box(hover, "hoverbox")}
          {hover && selection && (hover[0] !== selection[0] || hover[1] !== selection[1]) && box(hover, "hoverbox")}
          {live && box(live, "selbox live")}
          {selection && !live && box(selection, "selbox")}
          <div className="playhead" style={{ left: x(playhead) }} />
        </div>
      </div>
    </div>
    </div>
  );
}

export const TIMELINE_LABEL_W = LABEL_W;
export const TIMELINE_ROW = ROW;
