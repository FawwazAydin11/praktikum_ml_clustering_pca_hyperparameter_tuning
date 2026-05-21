const COLORS = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#f97316", "#0891b2", "#db2777", "#4f46e5"];

function rng(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normal(rand) {
  const u = Math.max(rand(), 1e-9);
  const v = Math.max(rand(), 1e-9);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function makeBlobData() {
  const rand = rng(42);
  const centers = [
    [-2.3, -1.25],
    [1.9, -1.1],
    [-1.55, 1.85],
    [2.15, 1.7]
  ];

  const data = [];

  centers.forEach(([cx, cy], cluster) => {
    for (let i = 0; i < 28; i++) {
      data.push({
        id: `${cluster}-${i}`,
        x: cx + normal(rand) * 0.43,
        y: cy + normal(rand) * 0.43,
        trueCluster: cluster
      });
    }
  });

  return data;
}

function makePcaData() {
  const rand = rng(11);
  const centers = [
    [-2.25, 0.2],
    [0.22, -0.5],
    [1.62, 0.42]
  ];
  const names = ["Setosa", "Versicolor", "Virginica"];
  const data = [];

  centers.forEach(([cx, cy], cluster) => {
    for (let i = 0; i < 38; i++) {
      data.push({
        id: `${names[cluster]}-${i}`,
        pc1: cx + normal(rand) * (cluster === 0 ? 0.24 : 0.43),
        pc2: cy + normal(rand) * 0.34,
        label: names[cluster],
        cluster
      });
    }
  });

  return data;
}

const DATA = makeBlobData();
const PCA_RAW = makePcaData();
const PCA_VARIANCE = [0.7296, 0.2285, 0.0367, 0.0052];

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function meanPoint(points) {
  const total = points.reduce(
    (acc, point) => {
      acc.x += point.x;
      acc.y += point.y;
      return acc;
    },
    { x: 0, y: 0 }
  );

  return {
    x: total.x / points.length,
    y: total.y / points.length
  };
}

function runKMeans(data, k, maxIter = 35) {
  const sorted = [...data].sort((a, b) => a.x - b.x || a.y - b.y);

  let centers = Array.from({ length: k }, (_, i) => {
    const index = Math.floor(((i + 0.5) * sorted.length) / k);
    return {
      x: sorted[index].x,
      y: sorted[index].y
    };
  });

  let labels = new Array(data.length).fill(0);

  for (let iter = 0; iter < maxIter; iter++) {
    let changed = false;

    data.forEach((point, pointIndex) => {
      let bestIndex = 0;
      let bestDistance = Infinity;

      centers.forEach((center, centerIndex) => {
        const dist = Math.hypot(point.x - center.x, point.y - center.y);

        if (dist < bestDistance) {
          bestDistance = dist;
          bestIndex = centerIndex;
        }
      });

      if (labels[pointIndex] !== bestIndex) {
        changed = true;
      }

      labels[pointIndex] = bestIndex;
    });

    centers = centers.map((oldCenter, centerIndex) => {
      const clusterPoints = data.filter((_, index) => labels[index] === centerIndex);
      return clusterPoints.length ? meanPoint(clusterPoints) : oldCenter;
    });

    if (!changed) break;
  }

  const inertia = data.reduce((total, point, index) => {
    const center = centers[labels[index]];
    return total + Math.pow(Math.hypot(point.x - center.x, point.y - center.y), 2);
  }, 0);

  return { labels, centers, inertia };
}

function runDBSCAN(data, eps, minSamples) {
  const labels = new Array(data.length).fill(undefined);
  const visited = new Array(data.length).fill(false);
  let clusterId = 0;

  function neighbors(index) {
    const result = [];

    data.forEach((point, otherIndex) => {
      if (distance(data[index], point) <= eps) {
        result.push(otherIndex);
      }
    });

    return result;
  }

  function expandCluster(index, neighborIndexes, currentCluster) {
    labels[index] = currentCluster;
    const queue = [...neighborIndexes];

    while (queue.length > 0) {
      const current = queue.shift();

      if (!visited[current]) {
        visited[current] = true;
        const currentNeighbors = neighbors(current);

        if (currentNeighbors.length >= minSamples) {
          currentNeighbors.forEach((neighbor) => {
            if (!queue.includes(neighbor)) {
              queue.push(neighbor);
            }
          });
        }
      }

      if (labels[current] === undefined || labels[current] === -1) {
        labels[current] = currentCluster;
      }
    }
  }

  data.forEach((_, index) => {
    if (visited[index]) return;

    visited[index] = true;
    const neighborIndexes = neighbors(index);

    if (neighborIndexes.length < minSamples) {
      labels[index] = -1;
    } else {
      expandCluster(index, neighborIndexes, clusterId);
      clusterId += 1;
    }
  });

  const coreIndexes = data
    .map((_, index) => ({
      index,
      count: neighbors(index).length
    }))
    .filter((item) => item.count >= minSamples)
    .map((item) => item.index);

  return {
    labels,
    clusterCount: clusterId,
    noiseCount: labels.filter((label) => label === -1).length,
    coreIndexes
  };
}

function clusterDistance(clusterA, clusterB, linkage) {
  const distances = [];

  clusterA.points.forEach((a) => {
    clusterB.points.forEach((b) => {
      distances.push(distance(a, b));
    });
  });

  if (linkage === "single") return Math.min(...distances);
  if (linkage === "complete") return Math.max(...distances);

  return distances.reduce((sum, value) => sum + value, 0) / distances.length;
}

function runHierarchical(data, targetClusters, linkage) {
  let clusters = data.map((point, index) => ({
    points: [point],
    indexes: [index]
  }));

  const mergeSteps = [];

  while (clusters.length > targetClusters) {
    let bestI = 0;
    let bestJ = 1;
    let bestDistance = Infinity;

    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        const dist = clusterDistance(clusters[i], clusters[j], linkage);

        if (dist < bestDistance) {
          bestDistance = dist;
          bestI = i;
          bestJ = j;
        }
      }
    }

    const merged = {
      points: [...clusters[bestI].points, ...clusters[bestJ].points],
      indexes: [...clusters[bestI].indexes, ...clusters[bestJ].indexes]
    };

    mergeSteps.push({
      distance: bestDistance,
      leftSize: clusters[bestI].points.length,
      rightSize: clusters[bestJ].points.length,
      mergedSize: merged.points.length,
      clusterCountAfterMerge: clusters.length - 1
    });

    clusters = clusters.filter((_, index) => index !== bestI && index !== bestJ);
    clusters.push(merged);
  }

  const labels = new Array(data.length).fill(0);

  clusters.forEach((cluster, clusterIndex) => {
    cluster.indexes.forEach((pointIndex) => {
      labels[pointIndex] = clusterIndex;
    });
  });

  return {
    labels,
    clusterCount: clusters.length,
    mergeSteps
  };
}

function silhouetteScore(data, labels, ignoreNoise = false) {
  const validIndexes = data
    .map((_, index) => index)
    .filter((index) => !ignoreNoise || labels[index] !== -1);

  const uniqueLabels = [...new Set(validIndexes.map((index) => labels[index]))];

  if (uniqueLabels.length < 2) return 0;

  const scores = validIndexes.map((index) => {
    const ownLabel = labels[index];
    const sameCluster = validIndexes.filter((other) => labels[other] === ownLabel && other !== index);

    const a = sameCluster.length
      ? sameCluster.reduce((sum, other) => sum + distance(data[index], data[other]), 0) / sameCluster.length
      : 0;

    const b = Math.min(
      ...uniqueLabels
        .filter((label) => label !== ownLabel)
        .map((label) => {
          const otherCluster = validIndexes.filter((other) => labels[other] === label);
          return otherCluster.reduce((sum, other) => sum + distance(data[index], data[other]), 0) / otherCluster.length;
        })
    );

    return (b - a) / Math.max(a, b);
  });

  return scores.reduce((sum, value) => sum + value, 0) / scores.length;
}

function getBounds(data) {
  const xs = data.map((point) => point.x);
  const ys = data.map((point) => point.y);

  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys)
  };
}

function project(point, bounds, width, height, padding = 42) {
  const usableWidth = width - padding * 2;
  const usableHeight = height - padding * 2;

  return {
    sx: padding + ((point.x - bounds.minX) / (bounds.maxX - bounds.minX || 1)) * usableWidth,
    sy: height - padding - ((point.y - bounds.minY) / (bounds.maxY - bounds.minY || 1)) * usableHeight
  };
}

function drawGrid(ctx, width, height) {
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;

  for (let x = 52; x < width; x += 72) {
    ctx.beginPath();
    ctx.moveTo(x, 28);
    ctx.lineTo(x, height - 42);
    ctx.stroke();
  }

  for (let y = 48; y < height; y += 62) {
    ctx.beginPath();
    ctx.moveTo(38, y);
    ctx.lineTo(width - 28, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "#cbd5e1";
  ctx.beginPath();
  ctx.moveTo(38, height - 42);
  ctx.lineTo(width - 28, height - 42);
  ctx.moveTo(38, 28);
  ctx.lineTo(38, height - 42);
  ctx.stroke();
}

function drawLegend(ctx, width, height, labels) {
  const unique = [...new Set(labels)].sort((a, b) => a - b).slice(0, 8);
  let x = 48;
  const y = height - 18;

  unique.forEach((label) => {
    const text = label === -1 ? "Noise" : `C${label + 1}`;
    const color = label === -1 ? "#64748b" : COLORS[Math.abs(label) % COLORS.length];

    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    ctx.fillStyle = "#475569";
    ctx.font = "bold 12px Inter, sans-serif";
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    ctx.fillText(text, x + 10, y);

    x += 68;
  });
}

function drawScatter(canvasId, data, labels, options = {}) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const bounds = getBounds(options.centers ? [...data, ...options.centers] : data);

  ctx.clearRect(0, 0, width, height);

  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, "#ffffff");
  gradient.addColorStop(1, "#f8fafc");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  drawGrid(ctx, width, height);

  if (options.centerLines && options.centers) {
    ctx.save();
    ctx.strokeStyle = "rgba(15, 23, 42, 0.1)";
    ctx.lineWidth = 1;

    data.forEach((point, index) => {
      if (index % 3 !== 0) return;

      const center = options.centers[labels[index]];
      const a = project(point, bounds, width, height);
      const b = project(center, bounds, width, height);

      ctx.beginPath();
      ctx.moveTo(a.sx, a.sy);
      ctx.lineTo(b.sx, b.sy);
      ctx.stroke();
    });

    ctx.restore();
  }

  if (options.eps && options.epsPointIndex !== undefined && options.epsPointIndex >= 0) {
    const point = data[options.epsPointIndex];
    const center = project(point, bounds, width, height);
    const edge = project({ x: point.x + options.eps, y: point.y }, bounds, width, height);
    const radius = Math.abs(edge.sx - center.sx);

    ctx.save();
    ctx.beginPath();
    ctx.arc(center.sx, center.sy, radius, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(249, 115, 22, 0.08)";
    ctx.fill();
    ctx.strokeStyle = "rgba(249, 115, 22, 0.58)";
    ctx.setLineDash([8, 7]);
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#9a3412";
    ctx.font = "bold 12px Inter, sans-serif";
    ctx.fillText("radius eps", center.sx + radius + 8, center.sy);
    ctx.restore();
  }

  data.forEach((point, index) => {
    const label = labels ? labels[index] : point.trueCluster ?? point.cluster ?? 0;
    const { sx, sy } = project(point, bounds, width, height);
    const color = label === -1 ? "#64748b" : COLORS[Math.abs(label) % COLORS.length];

    ctx.beginPath();
    ctx.arc(sx, sy, options.radius || 5.2, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.globalAlpha = label === -1 ? 0.58 : 0.88;
    ctx.fill();
    ctx.globalAlpha = 1;
  });

  if (options.centers) {
    options.centers.forEach((center) => {
      const { sx, sy } = project(center, bounds, width, height);

      ctx.beginPath();
      ctx.arc(sx, sy, 11, 0, Math.PI * 2);
      ctx.fillStyle = "#0f172a";
      ctx.fill();
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 13px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("C", sx, sy + 0.5);
    });
  }

  drawLegend(ctx, width, height, labels || data.map((point) => point.cluster ?? point.trueCluster ?? 0));
}

function projectToRect(point, bounds, rect, padding = 26) {
  const usableWidth = rect.w - padding * 2;
  const usableHeight = rect.h - padding * 2;

  return {
    sx: rect.x + padding + ((point.x - bounds.minX) / (bounds.maxX - bounds.minX || 1)) * usableWidth,
    sy: rect.y + rect.h - padding - ((point.y - bounds.minY) / (bounds.maxY - bounds.minY || 1)) * usableHeight
  };
}

function drawMiniGrid(ctx, rect) {
  ctx.save();
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 1;

  for (let x = rect.x + 42; x < rect.x + rect.w - 10; x += 60) {
    ctx.beginPath();
    ctx.moveTo(x, rect.y + 24);
    ctx.lineTo(x, rect.y + rect.h - 32);
    ctx.stroke();
  }

  for (let y = rect.y + 42; y < rect.y + rect.h - 20; y += 56) {
    ctx.beginPath();
    ctx.moveTo(rect.x + 24, y);
    ctx.lineTo(rect.x + rect.w - 16, y);
    ctx.stroke();
  }

  ctx.restore();
}

function drawHierarchicalDemo(canvasId, data, labels, mergeSteps, targetClusters) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const left = { x: 0, y: 0, w: Math.floor(width * 0.64), h: height };
  const right = { x: left.w + 12, y: 0, w: width - left.w - 12, h: height };
  const bounds = getBounds(data);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(left.x, left.y, left.w, left.h);
  ctx.fillRect(right.x, right.y, right.w, right.h);

  drawMiniGrid(ctx, left);

  data.forEach((point, index) => {
    const label = labels[index];
    const { sx, sy } = projectToRect(point, bounds, left, 38);
    const color = COLORS[Math.abs(label) % COLORS.length];

    ctx.beginPath();
    ctx.arc(sx, sy, 5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.88;
    ctx.fill();
    ctx.globalAlpha = 1;
  });

  ctx.save();
  ctx.strokeStyle = "#e2e8f0";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(left.w + 6, 18);
  ctx.lineTo(left.w + 6, height - 18);
  ctx.stroke();
  ctx.restore();

  drawLegend(ctx, left.w, height, labels);

  const dendroX = right.x + 28;
  const dendroY = right.y + 58;
  const dendroW = right.w - 54;
  const dendroH = right.h - 116;
  const importantSteps = mergeSteps.slice(-34);
  const maxDistance = Math.max(...importantSteps.map((step) => step.distance), 1);

  ctx.fillStyle = "#0f172a";
  ctx.font = "900 15px Inter, sans-serif";
  ctx.textAlign = "left";
  ctx.fillText("Mini Dendrogram", right.x + 20, 30);

  ctx.fillStyle = "#64748b";
  ctx.font = "bold 11px Inter, sans-serif";
  ctx.fillText("Atas = gabungan makin besar", right.x + 20, 47);

  ctx.strokeStyle = "#cbd5e1";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(dendroX, dendroY + dendroH);
  ctx.lineTo(dendroX, dendroY);
  ctx.moveTo(dendroX, dendroY + dendroH);
  ctx.lineTo(dendroX + dendroW, dendroY + dendroH);
  ctx.stroke();

  importantSteps.forEach((step, localIndex) => {
    const y = dendroY + dendroH - (localIndex / Math.max(importantSteps.length - 1, 1)) * dendroH;
    const x2 = dendroX + 12 + (step.distance / maxDistance) * (dendroW - 20);
    const thickness = Math.min(9, 2 + Math.log2(step.mergedSize + 1));

    ctx.strokeStyle = "rgba(37, 99, 235, 0.65)";
    ctx.lineWidth = thickness;
    ctx.beginPath();
    ctx.moveTo(dendroX, y);
    ctx.lineTo(x2, y);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(x2, y, Math.max(3, thickness / 2), 0, Math.PI * 2);
    ctx.fillStyle = "#2563eb";
    ctx.fill();
  });

  const cutRatio = (targetClusters - 2) / 4;
  const cutY = dendroY + 18 + cutRatio * (dendroH - 36);

  ctx.save();
  ctx.strokeStyle = "#ef4444";
  ctx.lineWidth = 2;
  ctx.setLineDash([7, 7]);
  ctx.beginPath();
  ctx.moveTo(dendroX - 6, cutY);
  ctx.lineTo(dendroX + dendroW, cutY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "#ef4444";
  ctx.font = "900 12px Inter, sans-serif";
  ctx.fillText(`cut: ${targetClusters} cluster`, dendroX + 8, cutY - 8);
  ctx.restore();

  ctx.fillStyle = "#334155";
  ctx.font = "bold 12px Inter, sans-serif";
  ctx.fillText("Jarak merge", dendroX, right.h - 32);

  ctx.fillStyle = "#64748b";
  ctx.font = "11px Inter, sans-serif";
  ctx.fillText("Garis merah = potongan cluster", right.x + 20, right.h - 14);
}

function renderBars(containerId, values) {
  const container = document.getElementById(containerId);
  if (!container) return;

  container.innerHTML = "";
  const max = Math.max(...values.map((item) => item.value));

  values.forEach((item) => {
    const wrapper = document.createElement("div");
    wrapper.className = `bar-item ${item.active ? "active" : ""}`;

    const bar = document.createElement("div");
    bar.className = "bar";
    bar.style.height = `${Math.max(12, (item.value / max) * 118)}px`;

    const label = document.createElement("div");
    label.className = "bar-label";
    label.textContent = item.label;

    wrapper.appendChild(bar);
    wrapper.appendChild(label);
    container.appendChild(wrapper);
  });
}

function explainSilhouette(score) {
  if (score >= 0.7) {
    return "Silhouette tinggi: kelompok terlihat rapi dan cukup jauh dari kelompok lain.";
  }

  if (score >= 0.45) {
    return "Silhouette sedang: cluster sudah terbentuk, tetapi beberapa titik dekat dengan batas antar cluster.";
  }

  if (score >= 0.15) {
    return "Silhouette rendah: beberapa cluster mulai bercampur atau pembagiannya belum terlalu jelas.";
  }

  return "Silhouette sangat rendah: pembagian cluster kurang jelas atau terlalu dipaksakan.";
}

function updateHierarchical() {
  const k = Number(document.getElementById("hierarchicalK").value);
  const linkage = document.getElementById("linkageSelect").value;
  const result = runHierarchical(DATA, k, linkage);
  const score = silhouetteScore(DATA, result.labels);

  document.getElementById("hierarchicalKValue").textContent = k;
  document.getElementById("hierarchicalClusterCount").textContent = result.clusterCount;
  document.getElementById("hierarchicalSilhouette").textContent = score.toFixed(3);

  let linkageText = "Average memakai rata-rata jarak antar kelompok, jadi hasilnya biasanya paling seimbang.";

  if (linkage === "single") {
    linkageText = "Single memakai jarak terdekat antar kelompok. Jika ada satu titik yang menjembatani dua kelompok, cluster bisa terlihat menyambung.";
  }

  if (linkage === "complete") {
    linkageText = "Complete memakai jarak terjauh antar kelompok. Hasilnya cenderung lebih ketat.";
  }

  document.getElementById("hierarchicalGuide").textContent =
    `Sekarang data dipotong menjadi ${k} cluster. Di dendrogram, garis merah adalah posisi potongnya. Jika jumlah cluster dikurangi, beberapa warna akan menyatu. ${linkageText} ${explainSilhouette(score)}`;

  drawHierarchicalDemo("hierarchicalCanvas", DATA, result.labels, result.mergeSteps, k);
}

function updateKMeans() {
  const k = Number(document.getElementById("kmeansK").value);
  const result = runKMeans(DATA, k);
  const score = silhouetteScore(DATA, result.labels);

  document.getElementById("kmeansKValue").textContent = k;
  document.getElementById("kmeansClusterCount").textContent = k;
  document.getElementById("kmeansInertia").textContent = result.inertia.toFixed(1);
  document.getElementById("kmeansSilhouette").textContent = score.toFixed(3);

  let kMeaning = "Nilai K ini cukup nyambung dengan pola data karena titik-titik memang terlihat punya beberapa kelompok alami.";

  if (k <= 2) {
    kMeaning = "K masih kecil, jadi dua area yang sebenarnya berbeda bisa dipaksa bergabung.";
  }

  if (k >= 7) {
    kMeaning = "K sudah besar, jadi satu kelompok alami bisa terpecah menjadi beberapa warna kecil.";
  }

  document.getElementById("kmeansGuide").textContent =
    `Warna adalah cluster, titik hitam C adalah centroid. Garis tipis menunjukkan contoh jarak titik ke centroid. Dengan K = ${k}, data dipaksa menjadi ${k} kelompok. ${kMeaning} Inertia = ${result.inertia.toFixed(1)}. ${explainSilhouette(score)}`;

  drawScatter("kmeansCanvas", DATA, result.labels, {
    centers: result.centers,
    centerLines: true
  });

  const bars = [];

  for (let kk = 2; kk <= 8; kk++) {
    bars.push({
      label: `K=${kk}`,
      value: runKMeans(DATA, kk).inertia,
      active: kk === k
    });
  }

  renderBars("elbowChart", bars);
}

function updateDBSCAN() {
  const eps = Number(document.getElementById("epsSlider").value);
  const minSamples = Number(document.getElementById("minSamplesSlider").value);
  const result = runDBSCAN(DATA, eps, minSamples);
  const score = result.clusterCount >= 2 ? silhouetteScore(DATA, result.labels, true) : 0;
  const epsPointIndex = result.coreIndexes[0] ?? DATA.findIndex((_, index) => result.labels[index] !== -1);

  document.getElementById("epsValue").textContent = eps.toFixed(2);
  document.getElementById("minSamplesValue").textContent = minSamples;
  document.getElementById("dbscanClusterCount").textContent = result.clusterCount;
  document.getElementById("dbscanNoiseCount").textContent = result.noiseCount;
  document.getElementById("dbscanSilhouette").textContent = score.toFixed(3);

  let meaning = "Parameter ini cukup seimbang: cluster terbentuk dan noise masih bisa diamati.";

  if (result.noiseCount > DATA.length * 0.25) {
    meaning = "Noise cukup banyak. Biasanya eps terlalu kecil atau min_samples terlalu besar.";
  }

  if (result.clusterCount <= 1) {
    meaning = "Cluster terlalu sedikit. Biasanya eps terlalu besar sehingga area berbeda ikut menyatu, atau aturan kepadatan belum cocok.";
  }

  if (result.clusterCount > 5) {
    meaning = "Cluster cukup banyak. Biasanya eps terlalu kecil sehingga kelompok besar terpecah.";
  }

  document.getElementById("dbscanGuide").textContent =
    `Lingkaran putus-putus menunjukkan contoh radius eps. Titik dalam radius itu dianggap tetangga. Dengan eps = ${eps.toFixed(2)} dan min_samples = ${minSamples}, terbentuk ${result.clusterCount} cluster dan ${result.noiseCount} noise. ${meaning}`;

  drawScatter("dbscanCanvas", DATA, result.labels, {
    eps,
    epsPointIndex
  });
}

function makePcaPlotData(components) {
  if (components === 1) {
    return PCA_RAW.map((item, index) => ({
      id: item.id,
      x: item.pc1,
      y: (index % 7 - 3) * 0.035,
      cluster: item.cluster,
      label: item.label
    }));
  }

  return PCA_RAW.map((item) => ({
    id: item.id,
    x: item.pc1,
    y: item.pc2,
    cluster: item.cluster,
    label: item.label
  }));
}

function updatePCA() {
  const components = Number(document.getElementById("pcaComponentSlider").value);
  const total = PCA_VARIANCE.slice(0, components).reduce((sum, value) => sum + value, 0);
  const plotData = makePcaPlotData(components);

  document.getElementById("pcaComponentValue").textContent = components;
  document.getElementById("pcaVariance").textContent = `${(total * 100).toFixed(2)}%`;

  let meaning = "Dengan 2 komponen, data bisa divisualisasikan sebagai grafik 2D dan sebagian besar informasi masih ikut terbawa.";

  if (components === 1) {
    meaning = "Dengan 1 komponen, titik hampir menjadi satu garis. Ini menunjukkan data sangat diringkas, tetapi informasi yang hilang lebih banyak.";
  }

  if (components >= 3) {
    meaning = "Dengan 3 atau 4 komponen, informasi makin banyak dipertahankan. Namun grafik tetap menampilkan PC1 dan PC2 karena layar hanya 2D.";
  }

  document.getElementById("pcaGuide").textContent =
    `Slider komponen memengaruhi banyaknya informasi yang disimpan. Saat ini ${components} komponen mempertahankan sekitar ${(total * 100).toFixed(2)}% informasi. ${meaning}`;

  drawScatter("pcaCanvas", plotData, plotData.map((item) => item.cluster), {
    radius: 5.5
  });

  renderBars(
    "varianceChart",
    PCA_VARIANCE.map((value, index) => ({
      label: `PC${index + 1}`,
      value,
      active: index < components
    }))
  );
}

function initHeroCanvas() {
  const result = runKMeans(DATA, 4);

  drawScatter("heroCanvas", DATA, result.labels, {
    centers: result.centers,
    radius: 4.8
  });
}

function bindEvents() {
  document.getElementById("hierarchicalK").addEventListener("input", updateHierarchical);
  document.getElementById("linkageSelect").addEventListener("change", updateHierarchical);
  document.getElementById("kmeansK").addEventListener("input", updateKMeans);
  document.getElementById("epsSlider").addEventListener("input", updateDBSCAN);
  document.getElementById("minSamplesSlider").addEventListener("input", updateDBSCAN);
  document.getElementById("pcaComponentSlider").addEventListener("input", updatePCA);
}

function init() {
  bindEvents();
  initHeroCanvas();
  updateHierarchical();
  updateKMeans();
  updateDBSCAN();
  updatePCA();
}

window.addEventListener("DOMContentLoaded", init);