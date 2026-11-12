/* XU Lab project page template — page.js
   Vanilla JS, no dependencies. Every feature is optional: the page is fully
   readable without this file (posters, first demo, data table, anchor links).

   Hooks (data attributes):
     video[data-src]          lazy video; loaded when near the viewport
     video[data-autoplay]     plays (muted, looped) while visible
     [data-demo]              demo switcher: .thumb buttons with data-src / data-poster
                              (+ optional data-caption shown in [data-demo-caption])
     [data-chart]             optional data-min-width (px, default 560) before the chart scrolls sideways
                              optional data-shadow (CSS colour) draws a hard drop shadow under each bar
     [data-chart]             bar chart rendered from the <table> inside it
     [data-spy]               side nav; links point at #ids of the cards
     [data-carousel]          ICL carousel: .chip buttons with data-human / data-robot (+ -poster)
*/
(function () {
  'use strict';

  var doc = document.documentElement;
  doc.classList.add('js');
  var motionQuery = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
  var reduceMotion = function () { return !!(motionQuery && motionQuery.matches); };
  var hasIO = 'IntersectionObserver' in window;
  var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };

  /* ------------------------------------------------------------ videos */
  function safePlay(video) {
    var p = video.play();
    if (p && p.catch) p.catch(function () { /* autoplay blocked: poster stays */ });
  }

  function loadVideo(video) {
    var src = video.getAttribute('data-src');
    if (src && video.getAttribute('src') !== src) {
      video.setAttribute('src', src);
      video.load();
    }
  }

  // Media fragments share the same HTTP resource; fragments are not sent
  // to the server. Explicit boundaries make looping consistent across browsers.
  function segment(video) {
    var source = video.getAttribute('data-src') || video.getAttribute('src') || '';
    var match = source.match(/#t=([\d.]+),([\d.]+)/);
    return match ? { start: Number(match[1]), end: Number(match[2]) } : null;
  }
  function setupSegment(video) {
    var looping = video.loop;
    function position() {
      var range = segment(video);
      if (!range) return;
      video.loop = false;
      if (video.currentTime < range.start || video.currentTime >= range.end) {
        video.currentTime = range.start;
      }
    }
    video.addEventListener('loadedmetadata', position);
    video.addEventListener('play', position);
    video.addEventListener('seeking', position);
    function boundary() {
      var range = segment(video);
      if (!range || video.seeking) return;
      if (video.currentTime >= range.end - 0.045) {
        video.pause();
        video.currentTime = range.start;
        if (looping && video._inView && !reduceMotion()) safePlay(video);
      }
    }
    video.addEventListener('timeupdate', boundary);
    video.addEventListener('ended', boundary);
    // Frame callbacks avoid showing a frame from the following scene.
    if (video.requestVideoFrameCallback) {
      var onFrame = function () { boundary(); video.requestVideoFrameCallback(onFrame); };
      video.requestVideoFrameCallback(onFrame);
    }
    if (segment(video)) video.loop = false;
  }

  function setupVideo(video) {
    // Autoplay clips stay silent; the full controlled video keeps its audio.
    video.muted = video.hasAttribute('muted') || video.hasAttribute('data-autoplay') || video.autoplay;
    video.setAttribute('playsinline', '');
    var auto = video.hasAttribute('data-autoplay') || video.autoplay;
    if (reduceMotion()) {
      video.autoplay = false;
      video.removeAttribute('autoplay');
      video.controls = true;
      video.pause();
      auto = false;
    }
    video._auto = auto;
    setupSegment(video);
    // Click toggles playback (native controls take over under reduced motion)
    video.addEventListener('click', function () {
      if (video.controls) return;
      if (video.paused) { loadVideo(video); safePlay(video); } else video.pause();
    });
  }

  var videos = $$('video');
  videos.forEach(setupVideo);

  if (hasIO) {
    var nearObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { loadVideo(e.target); nearObserver.unobserve(e.target); }
      });
    }, { rootMargin: '400px 0px' });
    var visObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        var v = e.target;
        v._inView = e.isIntersecting;
        if (!v._auto) return;
        if (e.isIntersecting) { loadVideo(v); safePlay(v); } else if (!v.paused) v.pause();
      });
    }, { threshold: 0.15 });
    videos.forEach(function (v) {
      if (v.hasAttribute('data-src')) nearObserver.observe(v);
      visObserver.observe(v);
    });
  } else {
    videos.forEach(function (v) { v._inView = true; loadVideo(v); if (v._auto) safePlay(v); });
  }

  function swapVideo(video, src, poster, label) {
    if (!video) return;
    if (video.tagName !== 'VIDEO') {
      // Placeholder frame (starter page): just show which item is selected
      if (label) video.textContent = label;
      return;
    }
    if (!src) return;
    if (poster) video.setAttribute('poster', poster);
    video.setAttribute('data-src', src);
    video.setAttribute('src', src);
    video.load();
    if (video._auto) safePlay(video);
  }

  /* ------------------------------------------------------------ demo switcher */
  $$('[data-demo]').forEach(function (root) {
    var player = root.querySelector('[data-demo-player]');
    var caption = root.querySelector('[data-demo-caption]');
    var thumbs = $$('.thumb', root);
    thumbs.forEach(function (btn) {
      btn.addEventListener('click', function () {
        thumbs.forEach(function (b) { b.setAttribute('aria-pressed', String(b === btn)); });
        var label = (btn.querySelector('.thumb__label') || btn).textContent.trim();
        swapVideo(player, btn.getAttribute('data-src'), btn.getAttribute('data-poster'), label);
        if (player && label) player.setAttribute('aria-label', label);
        if (caption && btn.hasAttribute('data-caption')) caption.textContent = btn.getAttribute('data-caption');
      });
    });
  });

  /* ------------------------------------------------------------ bar chart */
  var SVGNS = 'http://www.w3.org/2000/svg';
  function el(name, attrs, text) {
    var n = document.createElementNS(SVGNS, name);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    return n;
  }

  function readTable(table) {
    var heads = $$('thead th', table).slice(1);
    var series = heads.map(function (th, i) {
      return { name: th.textContent.trim(), color: th.getAttribute('data-color') || 'var(--c-series-' + ((i % 3) + 1) + ')' };
    });
    var cats = $$('tbody tr', table).map(function (tr) {
      var cells = $$('th, td', tr);
      return {
        name: cells[0].textContent.trim(),
        labelLines: (cells[0].getAttribute('data-chart-label') || cells[0].textContent.trim()).split('|'),
        color: cells[0].getAttribute('data-color'),
        colorLabel: cells[0].getAttribute('data-color-label') || cells[0].textContent.trim(),
        values: cells.slice(1).map(function (c) { return parseFloat(c.textContent); })
      };
    });
    return { series: series, cats: cats };
  }

  function renderChart(root) {
    var table = root.querySelector('table');
    var host = root.querySelector('.chart__scroll');
    var legend = root.querySelector('.chart__legend');
    if (!table || !host) return;
    var data = readTable(table);
    var categoryColors = data.series.length === 1 && data.cats.some(function (c) { return c.color; });
    function colorFor(c, si) {
      return categoryColors && c.color ? c.color : data.series[si].color;
    }
    var legendItems = data.series;
    if (categoryColors) {
      legendItems = [];
      data.cats.forEach(function (c) {
        if (!legendItems.some(function (s) { return s.name === c.colorLabel; })) {
          legendItems.push({ name: c.colorLabel, color: colorFor(c, 0) });
        }
      });
    }
    var max = parseFloat(root.getAttribute('data-max')) || 100;
    var unit = root.getAttribute('data-unit') || '%';
    var minW = parseFloat(root.getAttribute('data-min-width')) || 560;
    var shadow = root.getAttribute('data-shadow');   // e.g. "var(--c-bar-shadow)": paper-style hard shadow under bars
    var ticks = 4;

    // Legend
    if (legend) {
      legend.textContent = '';
      legendItems.forEach(function (s) {
        var li = document.createElement('li');
        var sw = document.createElement('span');
        sw.className = 'chart__swatch'; sw.style.background = s.color;
        if (shadow) sw.style.boxShadow = '1px 1px 0 ' + shadow;
        li.appendChild(sw); li.appendChild(document.createTextNode(s.name));
        legend.appendChild(li);
      });
    }

    var tip = document.createElement('div');
    tip.className = 'chart__tip'; tip.hidden = true;

    function draw() {
      var W = Math.max(host.clientWidth, minW);
      var extraLabelHeight = (Math.max.apply(null, data.cats.map(function (c) { return c.labelLines.length; })) - 1) * 14;
      var H = 240 + extraLabelHeight, m = { t: 20, r: 30, b: 30 + extraLabelHeight, l: 60 };
      var pw = W - m.l - m.r, ph = H - m.t - m.b;
      var svg = el('svg', { 'class': 'chart__svg', viewBox: '0 0 ' + W + ' ' + H, width: W, height: H, 'aria-hidden': 'true' });
      svg.style.width = W + 'px';
      var y = function (v) { return m.t + ph * (1 - v / max); };
      var band = pw / data.cats.length;
      var gap = band * 0.18, inner = band - 2 * gap, n = data.series.length;
      var bw = Math.min(44, (inner - 4 * (n - 1)) / n);
      var groupW = bw * n + 4 * (n - 1);

      var grid = el('g', { stroke: '#E4E2DA', 'stroke-dasharray': '3 3' });
      for (var i = 0; i <= ticks; i++) {
        var gy = y(max * i / ticks);
        grid.appendChild(el('line', { x1: m.l, x2: m.l + pw, y1: gy, y2: gy }));
      }
      data.cats.forEach(function (c, ci) {
        var cx = m.l + band * (ci + 0.5);
        grid.appendChild(el('line', { x1: cx, x2: cx, y1: m.t, y2: m.t + ph }));
      });
      grid.appendChild(el('line', { x1: m.l + pw, x2: m.l + pw, y1: m.t, y2: m.t + ph }));
      svg.appendChild(grid);

      var hover = el('rect', { x: 0, y: m.t, width: band, height: ph, fill: 'rgba(17,24,39,0.06)', opacity: 0 });
      svg.appendChild(hover);

      // Axes
      var axes = el('g', { stroke: '#666', fill: 'none' });
      axes.appendChild(el('line', { x1: m.l, x2: m.l, y1: m.t, y2: m.t + ph }));
      axes.appendChild(el('line', { x1: m.l, x2: m.l + pw, y1: m.t + ph, y2: m.t + ph }));
      for (var t = 0; t <= ticks; t++) {
        var ty = y(max * t / ticks);
        axes.appendChild(el('line', { x1: m.l - 6, x2: m.l, y1: ty, y2: ty }));
        svg.appendChild(el('text', { x: m.l - 9, y: ty, 'font-size': 12, fill: '#666', 'text-anchor': 'end', 'dominant-baseline': 'central' }, Math.round(max * t / ticks) + unit));
      }
      svg.appendChild(axes);

      data.cats.forEach(function (c, ci) {
        var x0 = m.l + band * ci + (band - groupW) / 2;
        c.labelLines.forEach(function (line, li) {
          svg.appendChild(el('text', { x: m.l + band * (ci + 0.5), y: m.t + ph + 16 + li * 14, 'font-size': 11, fill: '#666', 'text-anchor': 'middle' }, line));
        });
        c.values.forEach(function (v, si) {
          if (isNaN(v)) return;
          var bx = x0 + si * (bw + 4), by = y(v), bh = m.t + ph - by, r = Math.min(4, bh);
          var bar = 'M' + bx + ',' + (by + r) + 'a' + r + ',' + r + ' 0 0 1 ' + r + ',' + -r + 'h' + (bw - 2 * r) +
               'a' + r + ',' + r + ' 0 0 1 ' + r + ',' + r + 'V' + (m.t + ph) + 'H' + bx + 'Z';
          if (shadow) svg.appendChild(el('path', { d: bar, transform: 'translate(1.5,1.5)', style: 'fill:' + shadow }));
          svg.appendChild(el('path', { 'class': 'chart__bar', d: bar, style: 'fill:' + colorFor(c, si) }));
          svg.appendChild(el('text', { x: bx + bw / 2, y: by - 5, 'font-size': 10, fill: '#6A6A6A', 'text-anchor': 'middle' }, v + unit));
        });
        var hit = el('rect', { x: m.l + band * ci, y: m.t, width: band, height: ph, fill: 'transparent' });
        hit.addEventListener('mouseenter', function () {
          hover.setAttribute('x', m.l + band * ci); hover.setAttribute('opacity', 1);
          tip.innerHTML = '';
          var b = document.createElement('b'); b.textContent = c.name; tip.appendChild(b);
          data.series.forEach(function (s, si) {
            var row = document.createElement('span'); var dot = document.createElement('i');
            dot.style.background = colorFor(c, si); row.appendChild(dot);
            row.appendChild(document.createTextNode(s.name + ': ' + c.values[si] + unit));
            tip.appendChild(row);
          });
          tip.hidden = false;
          var left = m.l + band * (ci + 1) + 8;
          if (left + 170 > W) left = m.l + band * ci - 178;
          tip.style.left = left + 'px'; tip.style.top = m.t + 10 + 'px';
        });
        hit.addEventListener('mouseleave', function () { hover.setAttribute('opacity', 0); tip.hidden = true; });
        svg.appendChild(hit);
      });

      host.textContent = '';
      host.appendChild(svg);
      host.appendChild(tip);
    }

    draw();
    var lastW = host.clientWidth;
    window.addEventListener('resize', function () {
      if (host.clientWidth !== lastW) { lastW = host.clientWidth; draw(); }
    });
  }
  $$('[data-chart]').forEach(renderChart);

  /* ------------------------------------------------------------ side nav scroll spy */
  $$('[data-spy]').forEach(function (nav) {
    var links = $$('a[href^="#"]', nav);
    var targets = links.map(function (a) { return document.getElementById(a.getAttribute('href').slice(1)); });
    function update() {
      var marker = window.innerHeight * 0.4, current = 0;
      targets.forEach(function (t, i) { if (t && t.getBoundingClientRect().top <= marker) current = i; });
      links.forEach(function (a, i) {
        a.classList.toggle('is-active', i === current);
        if (i === current) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current');
      });
    }
    links.forEach(function (a, i) {
      a.addEventListener('click', function (ev) {
        var t = targets[i];
        if (!t) return;
        ev.preventDefault();
        window.scrollTo({ top: t.getBoundingClientRect().top + window.pageYOffset - 96, behavior: reduceMotion() ? 'auto' : 'smooth' });
        if (history.replaceState) history.replaceState(null, '', '#' + t.id);
      });
    });
    var ticking = false;
    window.addEventListener('scroll', function () {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(function () { ticking = false; update(); });
    }, { passive: true });
    window.addEventListener('resize', update);
    update();
  });

  /* ------------------------------------------------------------ ICL carousel */
  $$('[data-carousel]').forEach(function (root) {
    var chips = $$('.chip', root);
    var track = root.querySelector('.carousel__track');
    var human = root.querySelector('[data-slot="human"]');
    var robot = root.querySelector('[data-slot="robot"]');
    var index = Math.max(0, chips.findIndex(function (c) { return c.getAttribute('aria-pressed') === 'true'; }));

    function select(i, scroll) {
      index = (i + chips.length) % chips.length;
      chips.forEach(function (c, ci) { c.setAttribute('aria-pressed', String(ci === index)); });
      var chip = chips[index];
      var name = chip.textContent.trim();
      swapVideo(human, chip.getAttribute('data-human'), chip.getAttribute('data-human-poster'), 'Demo video · ' + name);
      swapVideo(robot, chip.getAttribute('data-robot'), chip.getAttribute('data-robot-poster'), 'Result video · ' + name);
      if (scroll && track) {
        // Horizontal-only centring (scrollIntoView would also scroll the page)
        var left = chip.offsetLeft - (track.clientWidth - chip.offsetWidth) / 2;
        track.scrollTo({ left: left, behavior: reduceMotion() ? 'auto' : 'smooth' });
      }
    }
    chips.forEach(function (c, ci) { c.addEventListener('click', function () { select(ci, true); }); });
    var prev = root.querySelector('[data-step="-1"]');
    var next = root.querySelector('[data-step="1"]');
    if (prev) prev.addEventListener('click', function () { select(index - 1, true); });
    if (next) next.addEventListener('click', function () { select(index + 1, true); });
    root.addEventListener('keydown', function (e) {
      if (e.target.classList && e.target.classList.contains('chip')) {
        if (e.key === 'ArrowRight') { select(index + 1, true); chips[index].focus(); }
        if (e.key === 'ArrowLeft') { select(index - 1, true); chips[index].focus(); }
      }
    });
  });

  /* ------------------------------------------------------------ reduced-motion changes at runtime */
  if (motionQuery && motionQuery.addEventListener) {
    motionQuery.addEventListener('change', function () {
      videos.forEach(function (v) {
        if (reduceMotion()) { v.pause(); v.controls = true; v._auto = false; }
      });
    });
  }

  // BibTeX copy button.
  document.querySelectorAll('[data-copy]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var src = document.querySelector(btn.getAttribute('data-copy'));
      if (!src) return;
      var done = function (ok) { btn.textContent = ok ? 'Copied' : 'Select to copy'; setTimeout(function () { btn.textContent = 'Copy'; }, 1600); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(src.textContent.trim()).then(function () { done(true); }, function () { done(false); });
      else done(false);
    });
  });
})();
