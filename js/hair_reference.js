// Local hairstyle gallery: modal browser + compact node UI. No model or generation.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const BASE = "/gallery/hair-gallery";
const imageUrl = (item, thumb = false) => api.apiURL(`${BASE}/image/${encodeURIComponent(item.id)}${thumb ? "?kind=thumb" : ""}`);
const el = (tag, cls = "", text = "") => { const e = document.createElement(tag); e.className = cls; e.textContent = text; return e; };
const button = (text, action, cls = "") => { const b = el("button", cls, text); b.type = "button"; b.onclick = action; return b; };

function styles() {
    for (const [id, file] of [["ref-pose-style", "pose_reference"], ["hair-reference-css", "hair_reference"]]) {
        if (document.getElementById(id)) continue;
        const link = el("link"); link.id = id; link.rel = "stylesheet";
        link.href = new URL(`./${file}.css`, import.meta.url).href; document.head.append(link);
    }
}

let pendingCatalog;
async function catalog(reload = false) {
    if (reload) pendingCatalog = null;
    if (!pendingCatalog) pendingCatalog = api.fetchApi(`${BASE}/catalog${reload ? "?reload=1" : ""}`).then(async response => {
        const data = await response.json();
        if (!response.ok) throw Error(data.error || "发型目录读取失败");
        if (data.version !== 1 || !data.items?.length) throw Error("发型目录格式无效");
        return data;
    }).catch(error => { pendingCatalog = null; throw error; });
    return pendingCatalog;
}

export function openHairGallery(selectedId, _language, onSelect) {
    styles();
    const dialog = el("dialog", "ref-pose ref-hair"), header = el("header");
    header.append(el("h2", "", "💇 本地发型图库"), button("重新读取目录", () => load(true)), button("关闭", () => dialog.close()));
    const filters = el("section", "dp-filters"), row = el("div", "dp-row");
    const search = el("input"); search.placeholder = "搜索发型编号（/）"; search.setAttribute("aria-label", "搜索发型");
    const groupSelect = el("select"); groupSelect.setAttribute("aria-label", "素材分组");
    groupSelect.add(new Option("全部分组", "")); groupSelect.add(new Option("图库", "图库")); groupSelect.add(new Option("编号库", "编号库"));
    const count = el("span", "dp-count"); count.setAttribute("role", "status");
    row.append(search, button("重置筛选", () => { search.value = ""; groupSelect.value = ""; filter(); }), count);
    filters.append(row, groupSelect);
    const main = el("main", "dp-main"), list = el("section", "dp-list"), detail = el("aside", "dp-detail");
    list.setAttribute("aria-label", "发型列表"); detail.setAttribute("aria-label", "发型详情"); detail.tabIndex = 0;
    main.append(list, detail);
    const footer = el("footer"), status = el("span", "dp-status"), apply = button("请先选择发型", applySelection, "dp-primary");
    status.setAttribute("role", "status"); apply.disabled = true; footer.append(status, apply);
    dialog.append(header, filters, main, footer);
    let data, matches = [], shown = 30, active, busy = false, epoch = 0, loadEpoch = 0;
    dialog.addEventListener("close", () => { epoch++; loadEpoch++; dialog.remove(); });
    dialog.onkeydown = event => {
        event.stopPropagation(); if (event.target.closest("input,textarea,select")) return;
        if (event.key === "/") { event.preventDefault(); search.focus(); }
        if (["ArrowLeft", "ArrowRight"].includes(event.key)) { event.preventDefault(); step(event.key === "ArrowRight" ? 1 : -1); }
    };
    search.oninput = groupSelect.onchange = filter;
    document.body.append(dialog); dialog.showModal(); load();

    async function load(reload = false) {
        const token = ++loadEpoch; list.replaceChildren(el("p", "", "正在读取本地发型目录…"));
        try {
            const value = await catalog(reload); if (!dialog.isConnected || token !== loadEpoch) return;
            data = value; filter();
            const item = data.items.find(i => i.id === (active?.id || selectedId));
            if (item) choose(item); else detail.replaceChildren(el("p", "", "点击发型查看大图，选好后点「使用这张发型」。"));
        } catch (error) {
            if (dialog.isConnected && token === loadEpoch) list.replaceChildren(el("p", "dp-error", error.message), button("重试读取目录", () => load(true)));
        }
    }
    function filter() {
        if (!data) return;
        const terms = search.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
        matches = data.items.filter(i => (!groupSelect.value || i.group === groupSelect.value) &&
            terms.every(t => `${i.id} ${i.name_zh} ${i.group} ${i.index}`.toLowerCase().includes(t)));
        shown = 30; count.textContent = `${matches.length} 张`; list.scrollTop = 0; drawCards();
    }
    function drawCards() {
        list.replaceChildren(); const grid = el("div", "dp-grid");
        for (const item of matches.slice(0, shown)) {
            const card = button("", () => choose(item), "dp-card dh-card"); card.dataset.id = item.id;
            card.setAttribute("aria-label", `${item.name_zh} ${item.id}`);
            const image = el("img"); image.src = imageUrl(item, true); image.loading = "lazy"; image.decoding = "async";
            image.alt = item.name_zh; image.onerror = () => { image.alt = "缩略图读取失败，点击查看原图"; };
            card.append(image, el("strong", "", item.name_zh), el("small", "", item.id));
            grid.append(card);
        }
        list.append(grid);
        if (!matches.length) list.append(el("p", "", "没有匹配的发型，请调整筛选或关键词。"));
        if (matches.length > shown) list.append(button(`加载更多（剩余 ${matches.length - shown}）`, () => { const scroll = list.scrollTop; shown += 30; drawCards(); list.scrollTop = scroll; }));
        mark();
    }
    function mark() { list.querySelectorAll(".dh-card").forEach(c => { c.classList.toggle("dp-selected", c.dataset.id === active?.id); c.setAttribute("aria-pressed", String(c.dataset.id === active?.id)); }); }
    function choose(item) {
        if (busy) return; active = item; status.textContent = ""; mark(); renderDetail(); detail.scrollTop = 0; detail.focus({ preventScroll: true });
        if (innerWidth <= 760) detail.scrollIntoView({ block: "start" });
    }
    function step(direction) {
        if (!matches.length || busy) return;
        const index = matches.findIndex(i => i.id === active?.id);
        choose(matches[index < 0 ? 0 : (index + direction + matches.length) % matches.length]);
    }
    function renderDetail() {
        const item = active; detail.replaceChildren();
        const image = el("img", "dp-detail-image"), hint = el("small", "", "正在加载原图…");
        image.src = imageUrl(item); image.alt = `${item.name_zh} 原图`;
        image.onload = () => { hint.textContent = "本地原图 · 输出为原尺寸 IMAGE"; };
        image.onerror = () => { hint.textContent = "原图读取失败，请重新读取目录"; };
        const nav = el("div", "dp-row");
        nav.append(button("上一张", () => step(-1)), button("下一张", () => step(1)),
            button("复制 ID", () => copy(item.id)), button("下载原图", () => download(item)));
        detail.append(el("h3", "", `${item.name_zh} · ${item.group}`), image, hint,
            el("small", "", `ID：${item.id} · 序号 ${item.index}/${data.count}`),
            el("p", "", "发型图为本地素材，仅作为参考图输出；不会调用任何模型，也不生成成片。"), nav);
        apply.disabled = busy; apply.textContent = `使用这张发型 · ${item.name_zh}`;
    }
    async function copy(text) { try { await navigator.clipboard.writeText(text); status.textContent = "已复制 ID"; } catch { status.textContent = "复制失败，请手动选中 ID。"; } }
    async function download(item) {
        try {
            const response = await api.fetchApi(imageUrl(item)); if (!response.ok) throw Error("原图读取失败");
            const url = URL.createObjectURL(await response.blob()), a = el("a"); a.href = url; a.download = `${item.id}.jpg`; a.click();
            setTimeout(() => URL.revokeObjectURL(url), 1500);
        } catch (e) { status.textContent = `下载失败：${e.message}`; }
    }
    async function applySelection() {
        if (!active || busy) return;
        const item = active, token = ++epoch; busy = true; apply.disabled = true; apply.textContent = "正在检查原图…";
        try {
            const image = new Image(); image.src = imageUrl(item); await image.decode();
            if (!dialog.isConnected || token !== epoch) return;
            await onSelect(item); dialog.close();
        } catch (e) { if (dialog.isConnected) status.textContent = `选择失败，原选择已保留：${e.message}`; }
        finally { busy = false; if (dialog.isConnected) renderDetail(); }
    }
    return dialog;
}

app.registerExtension({
    name: "HairReference",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "HairReference") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments); styles();
            const node = this, idWidget = node.widgets.find(w => w.name === "💇发型ID");
            if (!idWidget) return result;
            idWidget.hidden = true; idWidget.options ??= {}; idWidget.options.hidden = true; idWidget.computeSize = () => [0, -4];
            const input = idWidget.inputEl || idWidget.element || idWidget.domElement; if (input?.style) input.style.display = "none";
            const root = el("div", "ref-pose-node ref-hair-node"), heading = el("div", "dpn-heading");
            heading.append(el("strong", "", "💇 常用发型"), el("small", "", "点击缩略图快速选择"));
            const grid = el("div", "dpn-grid"), selected = el("div", "dpn-selected"), status = el("div", "dpn-message");
            status.setAttribute("role", "status");
            let data, alive = true, opened, busy = false, revision = 0;
            const random = button("🎲 随机发型", () => roll());
            const open = button("打开完整发型库  →", () => {
                if (busy || opened?.isConnected) return;
                const previous = idWidget.value;
                opened = openHairGallery(previous, null, item => {
                    if (idWidget.value !== previous) throw Error("节点参数已变化，请重新打开图库");
                    commit(item);
                });
            }, "dpn-open");
            root.append(heading, grid, selected, status, random, open,
                el("small", "dpn-footer", "本地发型素材 · 原尺寸 IMAGE 输出 · 不调用模型"));
            function commit(item) {
                if (!alive) throw Error("原节点已删除");
                node.graph?.beforeChange?.();
                try {
                    idWidget.value = item.id;
                    if (node.widgets_values) node.widgets_values[node.widgets.indexOf(idWidget)] = item.id;
                    idWidget.callback?.(item.id);
                } finally { node.graph?.afterChange?.(); }
                node.setDirtyCanvas?.(true, true); render();
            }
            function roll() {
                if (!data?.items?.length || busy) return;
                const pool = data.items.filter(i => i.id !== idWidget.value), list = pool.length ? pool : data.items;
                commit(list[Math.floor(Math.random() * list.length)]);
            }
            function render() {
                if (!alive) return; grid.replaceChildren(); selected.replaceChildren();
                const active = data?.items.find(i => i.id === idWidget.value);
                selected.append(el("span", "", active ? `已选：${active.name_zh} · ${active.id}` : idWidget.value ? "素材 ID 无效，请重新选择" : "请选择一张发型参考图"));
                if (data) {
                    let presets = data.items.slice(0, 6);
                    if (active && !presets.some(i => i.id === active.id)) presets = [active, ...presets.slice(0, 5)];
                    for (const item of presets) {
                        const card = button("", () => commit(item), "dpn-card");
                        card.title = item.id; card.setAttribute("aria-label", item.name_zh);
                        card.setAttribute("aria-pressed", String(item.id === active?.id));
                        const image = el("img"); image.src = imageUrl(item, true); image.alt = item.name_zh; image.draggable = false;
                        image.onerror = () => { image.alt = "缩略图读取失败"; };
                        card.append(image, el("span", "dpn-caption", item.name_zh));
                        if (item.id === active?.id) card.append(el("span", "dpn-check", "✓"));
                        grid.append(card);
                    }
                }
                random.disabled = open.disabled = busy || !data?.items?.length;
            }
            async function refresh(reload = false) {
                const token = ++revision; status.textContent = data ? "" : "正在读取本地发型目录…";
                try {
                    const value = await catalog(reload); if (!alive || token !== revision) return;
                    data = value; status.textContent = ""; render();
                } catch (e) { if (alive && token === revision) status.replaceChildren(el("span", "", e.message), button("重试", () => refresh(true))); }
            }
            for (const event of ["pointerdown", "keydown", "wheel"]) root.addEventListener(event, e => e.stopPropagation());
            const dom = node.addDOMWidget("hair_preview", "hair_preview", root, { serialize: false, getMinHeight: () => 430, getHeight: () => 430, hideOnZoom: false, hideInPanel: true });
            dom.options ??= {}; dom.options.minNodeSize = [360, 0]; dom.computeSize = w => [Math.max(360, w || 0), 430];
            node.setSize([430, 430]); refresh(); return result;
        };
    },
});
