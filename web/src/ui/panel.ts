// Side panel cards: each can be collapsed, pinned, or moved by dragging its grip (or with the arrow keys while the
// grip has focus). The layout is remembered per browser.
//
// From fly-addiction (which has it from PacFly, fly-games/apps/pacman/src/ui/panel.ts), with two changes: pinning a
// card moves it to the top of its column (and unpinning puts it back where it was), and a card is dragged with
// pointer events rather than the browser's drag and drop, which phones don't fire for a finger.
//
// On a desktop a card can also float: its float button (or a right-click on its grip) lifts it out of the panel into
// a window of its own, dragged by the grip anywhere on the screen and resized at its bottom-right corner. It keeps its
// place in the panel and goes back there when docked again (the same button, or another right-click).

const STORAGE_KEY = "flybrainfpv.panel";

interface Floating {
  /** left and top, px from the viewport's top-left corner */
  x: number;
  y: number;
  w: number;
}

interface PanelState {
  collapsed: string[];
  pinned: string | null;
  /** where the pinned card was (its index in its column) before it went to the top */
  pinnedFrom: number;
  /** column id -> section ids, top to bottom */
  columns: Record<string, string[]>;
  /** section id -> where it floats */
  floating: Record<string, Floating>;
}

const GRIP_SVG =
  '<svg viewBox="0 0 10 16" width="9" height="14" aria-hidden="true"><g fill="currentColor">' +
  '<circle cx="2.5" cy="3" r="1.3"/><circle cx="7.5" cy="3" r="1.3"/><circle cx="2.5" cy="8" r="1.3"/>' +
  '<circle cx="7.5" cy="8" r="1.3"/><circle cx="2.5" cy="13" r="1.3"/><circle cx="7.5" cy="13" r="1.3"/></g></svg>';
const PIN_SVG =
  '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><path fill="currentColor" d="M10.3 1.2l4.5 4.5-1.3 1.3-.9-.4-2.6 2.6.3 2.9-1.2 1.2-2.4-2.4-3.9 3.9H1.8v-.9l3.9-3.9-2.4-2.4 1.2-1.2 2.9.3 2.6-2.6-.4-.9z"/></svg>';
const FLOAT_SVG =
  '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><g fill="none" stroke="currentColor" stroke-width="1.5">' +
  '<path d="M7 2.5H3.5a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1V9"/><path d="M9.5 2.5h4v4M13.5 2.5 7.5 8.5"/></g></svg>';

/** px the pointer has to travel before a press on the grip becomes a drag */
const DRAG_START_PX = 4;
/** while dragging within this many px of the scrolling area's top or bottom edge, it scrolls */
const EDGE_PX = 48;
/** px a floating card moves per arrow key press on its grip */
const NUDGE_PX = 20;
/** where floating is offered: a wide window and a mouse (the same width the phone layout in styles.css starts under) */
const DESKTOP = "(min-width: 821px) and (pointer: fine)";

function load(ids: string[], columnIds: string[]): PanelState {
  try {
    const s = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null");
    if (s && Array.isArray(s.collapsed)) {
      const columns: Record<string, string[]> = {};
      for (const col of columnIds) {
        const list = s.columns?.[col];
        if (Array.isArray(list)) columns[col] = list.filter((id: unknown) => typeof id === "string" && ids.includes(id));
      }
      const floating: Record<string, Floating> = {};
      for (const [id, f] of Object.entries(s.floating ?? {})) {
        const { x, y, w } = (f ?? {}) as Partial<Floating>;
        if (ids.includes(id) && [x, y, w].every(Number.isFinite)) floating[id] = { x: x!, y: y!, w: w! };
      }
      return {
        collapsed: s.collapsed.filter((id: unknown) => typeof id === "string" && ids.includes(id)),
        pinned: typeof s.pinned === "string" && ids.includes(s.pinned) ? s.pinned : null,
        pinnedFrom: Number.isInteger(s.pinnedFrom) ? s.pinnedFrom : 0,
        columns,
        floating,
      };
    }
  } catch {
    // storage blocked or a corrupt value: start with the default layout
  }
  return { collapsed: [], pinned: null, pinnedFrom: 0, columns: {}, floating: {} };
}

function save(state: PanelState) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // not persisted, still works for this visit
  }
}

/** the nearest ancestor that scrolls vertically (the panel on a wide screen), or the page (on a phone) */
function scrollerOf(el: HTMLElement): HTMLElement | null {
  for (let p = el.parentElement; p; p = p.parentElement) {
    const oy = getComputedStyle(p).overflowY;
    if ((oy === "auto" || oy === "scroll") && p.scrollHeight > p.clientHeight) return p;
  }
  return null;
}


export interface PanelLayout {
  /** Every section back in its original column and order, expanded, unpinned and docked. */
  reset(): void;
}

export function setupPanel(panel: HTMLElement): PanelLayout {
  const columns = [...panel.querySelectorAll<HTMLElement>(".panel-col[data-column]")];
  const cardsOf = (col: HTMLElement) => [...col.querySelectorAll<HTMLElement>(":scope > .card[data-section]")];
  /** cards go before a column's trailing content that isn't a card (the Reset layout button) */
  const anchorOf = (col: HTMLElement) => col.querySelector<HTMLElement>(":scope > :not(.card):not(.drop-marker)");
  const layoutNow = () => Object.fromEntries(columns.map((col) => [col.dataset.column!, cardsOf(col).map((c) => c.dataset.section!)]));
  const defaults = layoutNow();
  const desktop = matchMedia(DESKTOP);

  const items = [...panel.querySelectorAll<HTMLElement>(".card[data-section]")].map((card) => {
    const h2 = card.querySelector("h2")!;
    const grip = document.createElement("button");
    grip.type = "button";
    grip.className = "card-grip";
    grip.innerHTML = GRIP_SVG;
    grip.setAttribute("aria-label", "Move section");
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "card-toggle";
    toggle.append(...h2.childNodes);
    const pin = document.createElement("button");
    pin.type = "button";
    pin.className = "card-pin";
    pin.innerHTML = PIN_SVG;
    const float = document.createElement("button");
    float.type = "button";
    float.className = "card-float";
    float.innerHTML = FLOAT_SVG;
    h2.append(grip, toggle, pin, float);
    return { id: card.dataset.section!, card, grip, toggle, pin, float };
  });
  const byId = new Map(items.map((it) => [it.id, it]));
  const state = load(
    items.map((it) => it.id),
    columns.map((col) => col.dataset.column!),
  );
  /** floating state is kept on a phone, but only shown on a desktop */
  const isFloating = (id: string) => desktop.matches && id in state.floating;
  /** the cards that sit in the column (not floating), top to bottom */
  const dockedOf = (col: HTMLElement) => cardsOf(col).filter((c) => !isFloating(c.dataset.section!));

  const place = (layout: Record<string, string[]>) => {
    for (const col of columns) {
      for (const id of layout[col.dataset.column!] ?? []) {
        const it = byId.get(id);
        if (it) col.insertBefore(it.card, anchorOf(col));
      }
    }
  };
  const record = () => {
    state.columns = layoutNow();
    save(state);
  };
  /** the pinned card first in its column, always */
  const pinToTop = () => {
    const card = state.pinned ? byId.get(state.pinned)?.card : null;
    const col = card?.parentElement;
    if (card && col) col.insertBefore(card, cardsOf(col)[0] ?? anchorOf(col));
  };
  /** the pinned card back to where it was before it was pinned */
  const unpin = () => {
    const card = state.pinned ? byId.get(state.pinned)?.card : null;
    const col = card?.parentElement;
    state.pinned = null;
    if (!card || !col) return;
    const rest = cardsOf(col).filter((c) => c !== card);
    col.insertBefore(card, rest[Math.min(state.pinnedFrom, rest.length)] ?? anchorOf(col));
  };

  /** a floating card where it was left, wholly on screen (a smaller window pushes it back in) */
  const position = (card: HTMLElement, f: Floating) => {
    card.style.width = `${Math.min(f.w, window.innerWidth)}px`;
    const x = Math.max(0, Math.min(f.x, window.innerWidth - card.offsetWidth));
    const y = Math.max(0, Math.min(f.y, window.innerHeight - card.offsetHeight));
    card.style.left = `${x}px`;
    card.style.top = `${y}px`;
    return { x, y };
  };
  /** the floating card last touched is drawn over the others */
  let front = 0;
  const raise = (card: HTMLElement) => {
    card.style.zIndex = String(30 + ++front);
  };

  const apply = () => {
    pinToTop();
    for (const { id, card, grip, toggle, pin, float } of items) {
      const collapsed = state.collapsed.includes(id);
      const pinned = state.pinned === id;
      const floating = isFloating(id);
      card.classList.toggle("collapsed", collapsed);
      card.classList.toggle("pinned", pinned);
      card.classList.toggle("floating", floating);
      toggle.setAttribute("aria-expanded", String(!collapsed));
      pin.setAttribute("aria-pressed", String(pinned));
      pin.title = pinned ? "Unpin" : "Pin to top";
      pin.setAttribute("aria-label", pinned ? "Unpin section" : "Pin section to top");
      float.setAttribute("aria-pressed", String(floating));
      float.title = floating ? "Dock back into the panel" : "Float anywhere on the screen";
      float.setAttribute("aria-label", floating ? "Dock section" : "Float section");
      grip.title = floating
        ? "Drag to move anywhere (or use the arrow keys); right-click to dock"
        : desktop.matches
          ? "Drag to move (or use the arrow keys); right-click to float"
          : "Drag to move (or use the arrow keys)";
      if (floating) position(card, state.floating[id]);
      else for (const p of ["width", "left", "top"]) card.style.removeProperty(p);
    }
  };

  /** float a card out of the panel, beside where it sat, or dock a floating one back into its place */
  const toggleFloat = (id: string) => {
    if (!desktop.matches) return;
    const card = byId.get(id)!.card;
    if (id in state.floating) {
      delete state.floating[id];
      card.style.removeProperty("z-index");
    } else {
      if (state.pinned === id) unpin();
      const r = card.getBoundingClientRect();
      state.floating[id] = { x: r.left - r.width - 24, y: r.top, w: r.width };
      raise(card);
    }
    record();
    apply();
  };

  // a floating card's width follows its resize handle
  const resized = new ResizeObserver((entries) => {
    for (const e of entries) {
      const card = e.target as HTMLElement;
      const f = state.floating[card.dataset.section!];
      if (!f || !card.classList.contains("floating") || drag) continue;
      const w = card.offsetWidth;
      if (w > 0 && w !== Math.round(f.w)) {
        f.w = w;
        save(state);
      }
    }
  });

  // ---- moving sections ------------------------------------------------------------
  // Pointer events on the grip, so a mouse, a finger and a pen all work; the grip has touch-action: none, so a
  // finger on it drags the card instead of scrolling the page.
  const marker = document.createElement("div");
  marker.className = "drop-marker";
  let drag: {
    card: HTMLElement;
    id: string;
    x0: number;
    y0: number;
    moving: boolean;
    y: number;
    x: number;
    /** a floating card: where the pointer holds it, from its top-left corner */
    hold: { dx: number; dy: number } | null;
  } | null = null;
  let scrollRaf = 0;

  /** put the marker where the card would land for a pointer at (x, y) */
  const placeMarker = (x: number, y: number) => {
    if (!drag) return;
    const col =
      columns.find((c) => {
        const r = c.getBoundingClientRect();
        return x >= r.left && x <= r.right;
      }) ?? (drag.card.parentElement as HTMLElement);
    // below the pinned card: it stays on top; floating cards aren't in the column to land next to
    const cards = dockedOf(col).filter((c) => c !== drag!.card && !(state.pinned && c.dataset.section === state.pinned));
    const next = cards.find((c) => {
      const r = c.getBoundingClientRect();
      return y < r.top + r.height / 2;
    });
    col.insertBefore(marker, next ?? anchorOf(col));
    for (const c of columns) c.classList.toggle("drop-target", c === col);
  };

  /** near an edge of what scrolls, keep scrolling, and keep the marker under the pointer */
  const autoScroll = () => {
    scrollRaf = 0;
    if (!drag?.moving) return;
    const box = scrollerOf(drag.card);
    const top = box ? box.getBoundingClientRect().top : 0;
    const bottom = box ? box.getBoundingClientRect().bottom : window.innerHeight;
    const speed = drag.y < top + EDGE_PX ? -(top + EDGE_PX - drag.y) : drag.y > bottom - EDGE_PX ? drag.y - (bottom - EDGE_PX) : 0;
    if (speed) {
      const dy = Math.max(-20, Math.min(20, speed * 0.4));
      if (box) box.scrollTop += dy;
      else window.scrollBy(0, dy);
      placeMarker(drag.x, drag.y);
    }
    scrollRaf = requestAnimationFrame(autoScroll);
  };

  /** a floating card follows the pointer; where it ends up is kept (on screen) */
  const moveFloating = () => {
    if (!drag?.hold) return;
    const f = state.floating[drag.id];
    const at = position(drag.card, { ...f, x: drag.x - drag.hold.dx, y: drag.y - drag.hold.dy });
    f.x = at.x;
    f.y = at.y;
  };

  const endDrag = (drop: boolean) => {
    cancelAnimationFrame(scrollRaf);
    if (drag?.hold) {
      if (drag.moving) save(state);
    } else if (drag?.moving && drop && marker.parentElement) {
      // dragging the pinned card somewhere is unpinning it there
      if (state.pinned === drag.id) state.pinned = null;
      marker.parentElement.insertBefore(drag.card, marker);
      record();
      apply();
    }
    marker.remove();
    for (const col of columns) col.classList.remove("drop-target");
    drag?.card.classList.remove("dragging");
    drag = null;
  };

  for (const { id, card, grip, toggle, pin, float } of items) {
    toggle.addEventListener("click", () => {
      state.collapsed = state.collapsed.includes(id) ? state.collapsed.filter((c) => c !== id) : [...state.collapsed, id];
      save(state);
      apply();
    });
    pin.addEventListener("click", () => {
      const was = state.pinned;
      unpin();
      if (was !== id) {
        const col = card.parentElement as HTMLElement;
        state.pinnedFrom = Math.max(0, cardsOf(col).indexOf(card));
        state.pinned = id;
        state.collapsed = state.collapsed.filter((c) => c !== id);
        // a pinned card sits in the panel (the pin button is hidden on a floating one, but not on a phone)
        delete state.floating[id];
      }
      record();
      apply();
    });
    float.addEventListener("click", () => toggleFloat(id));
    // the grip's own menu is of no use: a right-click floats or docks the card (a long press on a phone does nothing)
    grip.addEventListener("contextmenu", (e) => {
      e.preventDefault();
      toggleFloat(id);
    });
    card.addEventListener("pointerdown", () => card.classList.contains("floating") && raise(card), { capture: true });
    resized.observe(card);

    grip.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      grip.setPointerCapture(e.pointerId);
      const r = card.getBoundingClientRect();
      const hold = isFloating(id) ? { dx: e.clientX - r.left, dy: e.clientY - r.top } : null;
      drag = { card, id, x0: e.clientX, y0: e.clientY, x: e.clientX, y: e.clientY, moving: false, hold };
    });
    grip.addEventListener("pointermove", (e) => {
      if (!drag || drag.card !== card) return;
      drag.x = e.clientX;
      drag.y = e.clientY;
      if (!drag.moving) {
        if (Math.hypot(e.clientX - drag.x0, e.clientY - drag.y0) < DRAG_START_PX) return;
        drag.moving = true;
        if (drag.hold) {
          card.classList.add("moving");
        } else {
          card.classList.add("dragging");
          scrollRaf = requestAnimationFrame(autoScroll);
        }
      }
      if (drag.hold) moveFloating();
      else placeMarker(e.clientX, e.clientY);
    });
    const up = (drop: boolean) => {
      card.classList.remove("moving");
      endDrag(drop);
    };
    grip.addEventListener("pointerup", () => up(true));
    grip.addEventListener("pointercancel", () => up(false));

    grip.addEventListener("keydown", (e) => {
      if (isFloating(id)) {
        const f = state.floating[id];
        const step = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
        if (!step) return;
        e.preventDefault();
        const at = position(card, { ...f, x: f.x + step[0] * NUDGE_PX, y: f.y + step[1] * NUDGE_PX });
        f.x = at.x;
        f.y = at.y;
        save(state);
        return;
      }
      const col = card.parentElement as HTMLElement;
      const ci = columns.indexOf(col);
      if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        const cards = dockedOf(col);
        const i = cards.indexOf(card);
        const j = e.key === "ArrowUp" ? i - 1 : i + 1;
        if (j < 0 || j >= cards.length) return;
        // nothing goes above the pinned card, and moving the pinned card unpins it
        if (state.pinned && cards[j].dataset.section === state.pinned) return;
        if (state.pinned === id) state.pinned = null;
        col.insertBefore(card, e.key === "ArrowUp" ? cards[j] : cards[j].nextSibling);
      } else if ((e.key === "ArrowLeft" && ci > 0) || (e.key === "ArrowRight" && ci >= 0 && ci < columns.length - 1)) {
        if (state.pinned === id) state.pinned = null;
        const target = columns[ci + (e.key === "ArrowLeft" ? -1 : 1)];
        target.insertBefore(card, anchorOf(target));
      } else return;
      e.preventDefault();
      grip.focus();
      record();
      apply();
    });
  }

  place(state.columns);
  apply();
  // floating cards stay on screen as the window shrinks, and dock (for the view) in the phone layout
  window.addEventListener("resize", apply);
  desktop.addEventListener("change", apply);

  return {
    reset() {
      state.pinned = null;
      state.floating = {};
      for (const { card } of items) card.style.removeProperty("z-index");
      place(defaults);
      state.collapsed = [];
      record();
      apply();
    },
  };
}
