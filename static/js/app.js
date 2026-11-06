/**
 * CineMatch — Movie Search Frontend
 * Search a movie → get RBM-powered similar movie recommendations
 */

// ── Init ─────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    createParticles();
    loadStats();
    loadPopular();

    const input = document.getElementById("searchInput");
    const clearBtn = document.getElementById("clearBtn");

    // Debounced live search
    let debounceTimer;
    input.addEventListener("input", () => {
        const q = input.value.trim();
        clearBtn.classList.toggle("hidden", q === "");
        clearTimeout(debounceTimer);
        if (q.length < 2) { hideDropdown(); return; }
        debounceTimer = setTimeout(() => searchMovies(q), 230);
    });

    // Close dropdown when clicking outside
    document.addEventListener("click", (e) => {
        if (!e.target.closest(".search-wrapper")) hideDropdown();
    });

    // Keyboard nav
    input.addEventListener("keydown", handleDropdownKeys);

    // Clear button
    clearBtn.addEventListener("click", clearSearch);
});


// ── Particles ────────────────────────────────────────────────
function createParticles() {
    for (let i = 0; i < 28; i++) {
        const p = document.createElement("div");
        p.className = "particle";
        Object.assign(p.style, {
            left: Math.random() * 100 + "%",
            width:  (2 + Math.random() * 3) + "px",
            height: (2 + Math.random() * 3) + "px",
            animationDuration: (9 + Math.random() * 14) + "s",
            animationDelay:    (Math.random() * 12) + "s",
        });
        document.body.appendChild(p);
    }
}


// ── Stats ────────────────────────────────────────────────────
async function loadStats() {
    try {
        const d = await apiFetch("/api/stats");
        animateNum("statUsers",   d.stats.total_users);
        animateNum("statMovies",  d.stats.total_movies);
        animateNum("statRatings", d.stats.total_ratings);
        animateNum("statParams",  d.stats.model_params);
    } catch (_) {}
}

function animateNum(id, target) {
    const el = document.getElementById(id);
    const fmt = new Intl.NumberFormat();
    const start = performance.now(), dur = 1100;
    function tick(now) {
        const p = Math.min((now - start) / dur, 1);
        el.textContent = fmt.format(Math.round(target * (1 - Math.pow(1 - p, 3))));
        if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}


// ── Search ───────────────────────────────────────────────────
async function searchMovies(q) {
    try {
        const d = await apiFetch(`/api/search?q=${encodeURIComponent(q)}&limit=8`);
        renderDropdown(d.results);
    } catch (_) {
        hideDropdown();
    }
}

// Store current search results so we can reference them by index
let _dropdownMovies = [];

function renderDropdown(movies) {
    const dropdown = document.getElementById("dropdown");
    _dropdownMovies = movies;

    if (!movies.length) {
        dropdown.innerHTML = `<div class="dropdown-empty">No movies found for that title</div>`;
        dropdown.classList.remove("hidden");
        return;
    }

    dropdown.innerHTML = movies.map((m, i) => `
        <div class="dropdown-item" data-idx="${i}">
            <span class="di-icon">&#127902;</span>
            <div>
                <div class="di-title">${escHtml(m.title)}</div>
                <div class="di-genres">${escHtml(m.genres || "")}</div>
            </div>
        </div>
    `).join("");

    // Attach click listeners cleanly — no inline onclick, no escaping issues
    dropdown.querySelectorAll(".dropdown-item").forEach((el) => {
        el.addEventListener("click", () => {
            const idx = parseInt(el.dataset.idx, 10);
            const movie = _dropdownMovies[idx];
            if (movie) selectMovie(movie.movie_id, movie.title, movie.genres);
        });
    });

    dropdown.classList.remove("hidden");
}

function hideDropdown() {
    document.getElementById("dropdown").classList.add("hidden");
    _activeIdx = -1;
}

// Keyboard navigation
let _activeIdx = -1;
function handleDropdownKeys(e) {
    const items = document.querySelectorAll(".dropdown-item");
    if (!items.length) return;

    if (e.key === "ArrowDown") {
        e.preventDefault();
        _activeIdx = Math.min(_activeIdx + 1, items.length - 1);
    } else if (e.key === "ArrowUp") {
        e.preventDefault();
        _activeIdx = Math.max(_activeIdx - 1, 0);
    } else if (e.key === "Enter" && _activeIdx >= 0) {
        e.preventDefault();
        items[_activeIdx].click();
        return;
    } else if (e.key === "Escape") {
        hideDropdown(); return;
    } else {
        return;
    }
    items.forEach((el, i) =>
        el.style.background = i === _activeIdx ? "var(--bg-card-hover)" : ""
    );
}


// ── Select Movie → Load Similar ───────────────────────────────
async function selectMovie(movieId, title, genres) {
    document.getElementById("searchInput").value = title;
    document.getElementById("clearBtn").classList.remove("hidden");
    hideDropdown();

    // Show selected pill
    const pill = document.getElementById("selectedMovie");
    pill.innerHTML = `
        <span class="pill-icon">&#127916;</span>
        <span class="pill-label">Similar to</span>
        <strong>${escHtml(title)}</strong>
    `;

    // Show section with skeletons
    const section = document.getElementById("resultSection");
    const grid    = document.getElementById("similarGrid");
    section.classList.remove("hidden");
    grid.innerHTML = Array(8).fill('<div class="skeleton"></div>').join("");
    section.scrollIntoView({ behavior: "smooth", block: "start" });

    try {
        const d = await apiFetch(`/api/similar/${movieId}`);
        renderSimilar(d.similar);
    } catch (err) {
        grid.innerHTML = `<p style="color:var(--text-muted);padding:1rem">Could not load similar movies.</p>`;
        showToast("Error: " + err.message, true);
    }
}


// ── Render Similar ────────────────────────────────────────────
let _similarMovies = [];

function renderSimilar(movies) {
    const grid = document.getElementById("similarGrid");
    _similarMovies = movies;

    if (!movies.length) {
        grid.innerHTML = `<p style="color:var(--text-muted);padding:1rem">No similar movies found.</p>`;
        return;
    }

    grid.innerHTML = movies.map((m, i) => `
        <div class="movie-card" data-idx="${i}" style="animation-delay:${i * 0.055}s">
            <span class="card-badge badge-sim">${(m.similarity * 100).toFixed(0)}% match</span>
            <div class="card-title">${escHtml(m.title)}</div>
            <div class="card-genres">${genreTags(m.genres)}</div>
            <div class="card-meta">RBM cosine similarity <strong>${m.similarity.toFixed(3)}</strong></div>
        </div>
    `).join("");

    grid.querySelectorAll(".movie-card").forEach((el) => {
        el.addEventListener("click", () => {
            const idx = parseInt(el.dataset.idx, 10);
            const m = _similarMovies[idx];
            if (m) selectMovie(m.movie_id, m.title, m.genres);
        });
    });
}


// ── Clear Search ──────────────────────────────────────────────
function clearSearch() {
    document.getElementById("searchInput").value = "";
    document.getElementById("clearBtn").classList.add("hidden");
    hideDropdown();
    document.getElementById("resultSection").classList.add("hidden");
    document.getElementById("searchInput").focus();
}


// ── Popular Movies ────────────────────────────────────────────
let _popularMovies = [];

async function loadPopular() {
    try {
        const d = await apiFetch("/api/popular");
        _popularMovies = d.movies;
        const grid = document.getElementById("popularGrid");

        grid.innerHTML = d.movies.map((m, i) => `
            <div class="movie-card" data-idx="${i}" style="animation-delay:${i * 0.04}s">
                <span class="card-badge badge-rank">#${i + 1}</span>
                <div class="card-title">${escHtml(m.title)}</div>
                <div class="card-genres">${genreTags(m.genres)}</div>
                <div class="card-meta">
                    Score <strong>${m.score.toFixed(2)}</strong>
                    &nbsp;&bull;&nbsp; ${m.n_ratings.toLocaleString()} ratings
                </div>
                <div class="card-search-hint">&#128279; Find similar</div>
            </div>
        `).join("");

        grid.querySelectorAll(".movie-card").forEach((el) => {
            el.addEventListener("click", () => {
                const idx = parseInt(el.dataset.idx, 10);
                const m = _popularMovies[idx];
                if (m) selectMovie(m.movie_id, m.title, m.genres);
            });
        });

    } catch (_) {
        document.getElementById("popularGrid").innerHTML =
            `<p style="color:var(--text-muted)">Could not load trending movies.</p>`;
    }
}


// ── Helpers ───────────────────────────────────────────────────
async function apiFetch(endpoint) {
    const r = await fetch(endpoint);
    const d = await r.json();
    if (!d.success) throw new Error(d.error || "API error");
    return d;
}

function genreTags(genres) {
    if (!genres) return "";
    return genres.split(", ").filter(Boolean).slice(0, 3)
        .map(g => `<span class="genre-tag">${escHtml(g)}</span>`).join("");
}

function escHtml(str) {
    if (!str) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function showToast(msg, isError = false) {
    const t = document.getElementById("toast");
    t.textContent = msg;
    t.className = "toast show" + (isError ? " error" : "");
    setTimeout(() => { t.className = "toast hidden"; }, 3000);
}
