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
  const centers = [[-2.2, -1.3], [1.8, -1.1], [-1.4, 1.8], [2.0, 1.7]];
  const data = [];

  centers.forEach(([cx, cy], cluster) => {
    for (let i = 0; i < 26; i++) {
      data.push({
        id: `${cluster}-${i}`,
        x: cx + normal(rand) * 0.45,
        y: cy + normal(rand) * 0.45,
        trueCluster: cluster
      });
    }
  });

  return data;
}

function makePcaData() {
  const rand = rng(11);
  const centers = [[-2.25, 0.25], [0.25, -0.52], [1.65, 0.38]];
  const names = ["Setosa", "Versicolor", "Virginica"];
  const data = [];

  centers.forEach(([cx, cy], cluster) => {
    for (let i = 0; i < 38; i++) {
      data.push({
        id: `${names[cluster]}-${i}`,
        x: cx + normal(rand) * (cluster === 0 ? 0.26 : 0.45),
        y: cy + normal(rand) * 0.34,
        label: names[cluster],
        cluster
      });
    }
  });

  return data;
}

const DATA = makeBlobData();
const PCA_DATA = makePcaData();
const PCA_VARIANCE = [0.7296, 0.2285, 0.0367, 0.0052];

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function meanPoint(points) {
  const total = points.reduce((acc, p) => {
    acc.x += p.x;
    acc.y += p.y;
    return acc;
  }, { x: 0, y: 0 });

  return { x: total.x / points.length, y: total.y / points.length };
}

function runKMeans(data, k, maxIter = 30) {
  let centers = Array.from({ length: k }, (_, i) => ({
    x: data[Math.floor((i * data.length) / k)].x,
    y: data[Math.floor((i * data.length) / k)].y
  }));

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

      if (labels[pointIndex] !== bestIndex) changed = true;
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
      if (distance(data[index], point) <= eps) result.push(otherIndex);
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
            if (!queue.includes(neighbor)) queue.push(neighbor);
          });
        }
      }

      if (labels[current] === undefined || labels[current] === -1) labels[current] = currentCluster;
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
      clusterId++;
    }
  });

  return {
    labels,
    clusterCount: clusterId,
    noiseCount: labels.filter((label) => label === -1).length
  };
}

function clusterDistance(clusterA, clusterB, linkage) {
  const distances = [];

  clusterA.points.forEach((a) => {
    clusterB.points.forEach((b) => distances.push(distance(a, b)));
  });

  if (linkage === "single") return Math.min(...distances);
  if (linkage === "complete") return Math.max(...distances);
  return distances.reduce((sum, value) => sum + value, 0) / distances.length;
}

function runHierarchical(data, targetClusters, linkage) {
  let clusters = data.map((point, index) => ({ points: [point], indexes: [index] }));

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

    clusters = clusters.filter((_, index) => index !== bestI && index !== bestJ);
    clusters.push(merged);
  }

  const labels = new Array(data.length).fill(0);
  clusters.forEach((cluster, clusterIndex) => {
    cluster.indexes.forEach((pointIndex) => labels[pointIndex] = clusterIndex);
  });

  return { labels, clusterCount: clusters.length };
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
  const xs = data.map((p) => p.x);
  const ys = data.map((p) => p.y);
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
    sx: padding + ((point.x - bounds.minX) / (bounds.maxX - bounds.minX)) * usableWidth,
    sy: height - padding - ((point.y - bounds.minY) / (bounds.maxY - bounds.minY)) * usableHeight
  };
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

  drawLegend(ctx, width, height, labels || data.map((p) => p.cluster ?? p.trueCluster ?? 0));
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
  if (score >= 0.7) return "Nilai silhouette tinggi, berarti cluster terlihat cukup rapi dan jarak antar kelompok cukup jelas.";
  if (score >= 0.45) return "Nilai silhouette sedang, berarti cluster sudah terbentuk tetapi masih ada beberapa titik yang dekat dengan batas antar cluster.";
  if (score >= 0.15) return "Nilai silhouette rendah, berarti beberapa cluster mulai bercampur atau belum terlalu rapi.";
  return "Nilai silhouette sangat rendah, berarti pembagian cluster kurang jelas atau terlalu dipaksakan.";
}

function updateHierarchical() {
  const k = Number(document.getElementById("hierarchicalK").value);
  const linkage = document.getElementById("linkageSelect").value;
  const result = runHierarchical(DATA, k, linkage);
  const score = silhouetteScore(DATA, result.labels);

  document.getElementById("hierarchicalKValue").textContent = k;
  document.getElementById("hierarchicalClusterCount").textContent = result.clusterCount;
  document.getElementById("hierarchicalSilhouette").textContent = score.toFixed(3);

  let linkageText = "Average melihat rata-rata jarak antar kelompok, jadi hasilnya cenderung seimbang.";
  if (linkage === "single") linkageText = "Single melihat jarak terdekat antar kelompok. Kadang cluster bisa terlihat seperti rantai panjang.";
  if (linkage === "complete") linkageText = "Complete melihat jarak terjauh antar kelompok. Hasilnya biasanya lebih ketat dan tidak mudah menyatu.";

  document.getElementById("hierarchicalGuide").textContent =
    `Sekarang data dibagi menjadi ${k} cluster. ${linkageText} ${explainSilhouette(score)}`;

  drawScatter("hierarchicalCanvas", DATA, result.labels);
}

function updateKMeans() {
  const k = Number(document.getElementById("kmeansK").value);
  const result = runKMeans(DATA, k);
  const score = silhouetteScore(DATA, result.labels);

  document.getElementById("kmeansKValue").textContent = k;
  document.getElementById("kmeansClusterCount").textContent = k;
  document.getElementById("kmeansInertia").textContent = result.inertia.toFixed(1);
  document.getElementById("kmeansSilhouette").textContent = score.toFixed(3);

  let kMeaning = "Nilai K ini cukup masuk akal untuk dataset yang memang terlihat punya beberapa kelompok alami.";
  if (k <= 2) kMeaning = "K masih kecil, jadi beberapa kelompok yang sebenarnya berbeda kemungkinan digabung menjadi satu.";
  if (k >= 7) kMeaning = "K sudah besar, jadi beberapa kelompok bisa terpecah terlalu detail.";

  document.getElementById("kmeansGuide").textContent =
    `Dengan K = ${k}, data dipaksa menjadi ${k} kelompok. ${kMeaning} Inertia saat ini ${result.inertia.toFixed(1)}; makin kecil berarti titik makin dekat dengan centroid, tetapi K terlalu besar juga belum tentu lebih baik. ${explainSilhouette(score)}`;

  drawScatter("kmeansCanvas", DATA, result.labels, { centers: result.centers });

  const bars = [];
  for (let kk = 2; kk <= 8; kk++) {
    bars.push({ label: `K=${kk}`, value: runKMeans(DATA, kk).inertia, active: kk === k });
  }
  renderBars("elbowChart", bars);
}

function updateDBSCAN() {
  const eps = Number(document.getElementById("epsSlider").value);
  const minSamples = Number(document.getElementById("minSamplesSlider").value);
  const result = runDBSCAN(DATA, eps, minSamples);
  const score = result.clusterCount >= 2 ? silhouetteScore(DATA, result.labels, true) : 0;

  document.getElementById("epsValue").textContent = eps.toFixed(2);
  document.getElementById("minSamplesValue").textContent = minSamples;
  document.getElementById("dbscanClusterCount").textContent = result.clusterCount;
  document.getElementById("dbscanNoiseCount").textContent = result.noiseCount;
  document.getElementById("dbscanSilhouette").textContent = score.toFixed(3);

  let meaning = "Parameter ini cukup seimbang: cluster terbentuk dan noise masih bisa diamati.";
  if (result.noiseCount > DATA.length * 0.25) meaning = "Noise cukup banyak. Biasanya ini terjadi karena eps terlalu kecil atau min_samples terlalu ketat.";
  if (result.clusterCount <= 1) meaning = "Cluster terlalu sedikit. Bisa jadi eps terlalu besar sehingga kelompok menyatu, atau aturan kepadatan belum cocok.";
  if (result.clusterCount > 5) meaning = "Cluster cukup banyak. Bisa jadi eps terlalu kecil sehingga kelompok besar terpecah menjadi kelompok kecil.";

  document.getElementById("dbscanGuide").textContent =
    `Dengan eps = ${eps.toFixed(2)} dan min_samples = ${minSamples}, terbentuk ${result.clusterCount} cluster dan ${result.noiseCount} noise. ${meaning} Titik abu-abu berarti data yang tidak cukup dekat dengan kelompok mana pun.`;

  drawScatter("dbscanCanvas", DATA, result.labels);
}

function updatePCA() {
  const components = Number(document.getElementById("pcaComponentSlider").value);
  const total = PCA_VARIANCE.slice(0, components).reduce((sum, value) => sum + value, 0);

  document.getElementById("pcaComponentValue").textContent = components;
  document.getElementById("pcaVariance").textContent = `${(total * 100).toFixed(2)}%`;

  let meaning = "Jumlah komponen ini cukup umum dipakai untuk visualisasi karena bisa ditampilkan di grafik 2D.";
  if (components === 1) meaning = "Data sangat diringkas, tetapi informasi yang hilang lebih banyak. Cocok untuk melihat gambaran paling sederhana saja.";
  if (components >= 3) meaning = "Informasi yang disimpan makin besar, tetapi data tidak sesederhana visualisasi 2D.";

  document.getElementById("pcaGuide").textContent =
    `Dengan ${components} komponen, PCA mempertahankan sekitar ${(total * 100).toFixed(2)}% informasi. ${meaning} Grafik tetap menampilkan dua komponen pertama agar mudah dilihat.`;

  drawScatter("pcaCanvas", PCA_DATA, PCA_DATA.map((item) => item.cluster), { radius: 5.5 });

  renderBars("varianceChart", PCA_VARIANCE.map((value, index) => ({
    label: `PC${index + 1}`,
    value,
    active: index < components
  })));
}

function initHeroCanvas() {
  const result = runKMeans(DATA, 4);
  drawScatter("heroCanvas", DATA, result.labels, { centers: result.centers, radius: 4.8 });
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
