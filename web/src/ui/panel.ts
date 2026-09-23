// From fly-addiction (which has it from PacFly, fly-games/apps/pacman/src/ui/panel.ts), unchanged but for the storage key.
//
// Side panel cards: each can be collapsed, pinned to the top of its column, or moved by dragging its grip to another
// place or column (or with the arrow keys while the grip has focus). The layout is remembered per browser.

const STORAGE_KEY = "flybrainfpv.panel";

interface PanelState {
  collapsed: string[];
  pinned: string | null;
  /** column id -> section ids, top to bottom */
  columns: Record<string, string[]>;
}

const GRIP_SVG =
  '<svg viewBox="0 0 10 16" width="9" height="14" aria-hidden="true"><g fill="currentColor">' +
  '<circle cx="2.5" cy="3" r="1.3"/><circle cx="7.5" cy="3" r="1.3"/><circle cx="2.5" cy="8" r="1.3"/>' +
  '<circle cx="7.5" cy="8" r="1.3"/><circle cx="2.5" cy="13" r="1.3"/><circle cx="7.5" cy="13" r="1.3"/></g></svg>';
const PIN_SVG =
  '<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true"><path fill="currentColor" d="M10.3 1.2l4.5 4.5-1.3 1.3-.9-.4-2.6 2.6.3 2.9-1.2 1.2-2.4-2.4-3.9 3.9H1.8v-.9l3.9-3.9-2.4-2.4 1.2-1.2 2.9.3 2.6-2.6-.4-.9z"/></svg>';

function load(ids: string[], columnIds: string[]): PanelState {
  try {
    const s = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null");
    if (s && Array.isArray(s.collapsed)) {
      const columns: Record<string, string[]> = {};
      for (const col of columnIds) {
        const list = s.columns?.[col];
        if (Array.isArray(list)) columns[col] = list.filter((id: unknown) => typeof id === "string" && ids.includes(id));
      }
      return {
        collapsed: s.collapsed.filter((id: unknown) => typeof id === "string" && ids.includes(id)),
        pinned: typeof s.pinned === "string" && ids.includes(s.pinned) ? s.pinned : null,
        columns,
      };
    }
  } catch {
    // storage blocked or a corrupt value: start with the default layout
  }
  return { collapsed: [], pinned: null, columns: {} };
}

function save(state: PanelState) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // not persisted, still works for this visit
  }
}

export interface PanelLayout {
  /** Every section back in its original column and order, expanded and unpinned. */
  reset(): void;
}

export function setupPanel(panel: HTMLElement): PanelLayout {
  const columns = [...panel.querySelectorAll<HTMLElement>(".panel-col[data-column]")];
  const cardsOf = (col: HTMLElement) => [...col.querySelectorAll<HTMLElement>(":scope > .card[data-section]")];
  /** cards go before a column's trailing content that isn't a card (the credits) */
  const anchorOf = (col: HTMLElement) => col.querySelector<HTMLElement>(":scope > :not(.card):not(.drop-marker)");
  const layoutNow = () => Object.fromEntries(columns.map((col) => [col.dataset.column!, cardsOf(col).map((c) => c.dataset.section!)]));
  const defaults = layoutNow();

  const items = [...panel.querySelectorAll<HTMLElement>(".card[data-section]")].map((card) => {
    const h2 = card.querySelector("h2")!;
    const grip = document.createElement("button");
    grip.type = "button";
    grip.className = "card-grip";
    grip.innerHTML = GRIP_SVG;
    grip.title = "Drag to move (or use the arrow keys)";
    grip.setAttribute("aria-label", "Move section");
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "card-toggle";
    toggle.append(...h2.childNodes);
    const pin = document.createElement("button");
    pin.type = "button";
    pin.className = "card-pin";
    pin.innerHTML = PIN_SVG;
    h2.append(grip, toggle, pin);
    return { id: card.dataset.section!, card, grip, toggle, pin };
  });
  const byId = new Map(items.map((it) => [it.id, it]));
  const state = load(
    items.map((it) => it.id),
    columns.map((col) => col.dataset.column!),
  );

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

  const apply = () => {
    for (const { id, card, toggle, pin } of items) {
      const collapsed = state.collapsed.includes(id);
      const pinned = state.pinned === id;
      card.classList.toggle("collapsed", collapsed);
      card.classList.toggle("pinned", pinned);
      toggle.setAttribute("aria-expanded", String(!collapsed));
      pin.setAttribute("aria-pressed", String(pinned));
      pin.title = pinned ? "Unpin" : "Pin to top";
      pin.setAttribute("aria-label", pinned ? "Unpin section" : "Pin section to top");
    }
  };

  // ---- moving sections ------------------------------------------------------------
  // The card only becomes draggable while the grip is held, so dragging a brain view to rotate it still works.
  let dragged: HTMLElement | null = null;
  const marker = document.createElement("div");
  marker.className = "drop-marker";
  const endDrag = () => {
    marker.remove();
    for (const col of columns) col.classList.remove("drop-target");
    if (dragged) {
      dragged.draggable = false;
      dragged.classList.remove("dragging");
    }
    dragged = null;
  };

  for (const { id, card, grip, toggle, pin } of items) {
    toggle.addEventListener("click", () => {
      state.collapsed = state.collapsed.includes(id) ? state.collapsed.filter((c) => c !== id) : [...state.collapsed, id];
      save(state);
      apply();
    });
    pin.addEventListener("click", () => {
      state.pinned = state.pinned === id ? null : id;
      if (state.pinned) state.collapsed = state.collapsed.filter((c) => c !== id);
      save(state);
      apply();
    });

    grip.addEventListener("pointerdown", () => (card.draggable = true));
    grip.addEventListener("pointerup", () => {
      if (dragged !== card) card.draggable = false;
    });
    card.addEventListener("dragstart", (e) => {
      if (!card.draggable) return;
      dragged = card;
      e.dataTransfer?.setData("text/plain", id);
      if (e.dataTransfer) e.dataTransfer.effectAllowed = "move";
      // after the browser has taken its drag image
      requestAnimationFrame(() => card.classList.add("dragging"));
    });
    card.addEventListener("dragend", endDrag);

    grip.addEventListener("keydown", (e) => {
      const col = card.parentElement as HTMLElement;
      const ci = columns.indexOf(col);
      if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        const cards = cardsOf(col);
        const i = cards.indexOf(card);
        const j = e.key === "ArrowUp" ? i - 1 : i + 1;
        if (j < 0 || j >= cards.length) return;
        col.insertBefore(card, e.key === "ArrowUp" ? cards[j] : cards[j].nextSibling);
      } else if ((e.key === "ArrowLeft" && ci > 0) || (e.key === "ArrowRight" && ci >= 0 && ci < columns.length - 1)) {
        const target = columns[ci + (e.key === "ArrowLeft" ? -1 : 1)];
        target.insertBefore(card, anchorOf(target));
      } else return;
      e.preventDefault();
      grip.focus();
      record();
    });
  }

  for (const col of columns) {
    col.addEventListener("dragover", (e) => {
      if (!dragged) return;
      e.preventDefault();
      if (e.dataTransfer) e.dataTransfer.dropEffect = "move";
      const next = cardsOf(col).find((c) => {
        if (c === dragged) return false;
        const r = c.getBoundingClientRect();
        return e.clientY < r.top + r.height / 2;
      });
      col.insertBefore(marker, next ?? anchorOf(col));
      for (const c of columns) c.classList.toggle("drop-target", c === col);
    });
    col.addEventListener("drop", (e) => {
      if (!dragged || marker.parentElement !== col) return;
      e.preventDefault();
      col.insertBefore(dragged, marker);
      endDrag();
      record();
    });
  }

  place(state.columns);
  apply();

  return {
    reset() {
      place(defaults);
      state.collapsed = [];
      state.pinned = null;
      record();
      apply();
    },
  };
}
