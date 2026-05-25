function loadImage(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = (e) => {
      URL.revokeObjectURL(url);
      reject(e);
    };
    img.src = url;
  });
}

function sampleBackgroundColor(
  ctx: CanvasRenderingContext2D,
  img: HTMLImageElement
): string {
  // Sample the four corner pixels of the source image to decide background fill.
  // Most jewel photographs have a uniform matte black or white background.
  const probe = document.createElement("canvas");
  probe.width = img.width;
  probe.height = img.height;
  const pctx = probe.getContext("2d")!;
  pctx.drawImage(img, 0, 0);
  const pad = 4;
  const samples = [
    pctx.getImageData(pad, pad, 1, 1).data,
    pctx.getImageData(img.width - pad - 1, pad, 1, 1).data,
    pctx.getImageData(pad, img.height - pad - 1, 1, 1).data,
    pctx.getImageData(img.width - pad - 1, img.height - pad - 1, 1, 1).data,
  ];
  let r = 0;
  let g = 0;
  let b = 0;
  for (const s of samples) {
    r += s[0];
    g += s[1];
    b += s[2];
  }
  r = Math.round(r / samples.length);
  g = Math.round(g / samples.length);
  b = Math.round(b / samples.length);
  return `rgb(${r}, ${g}, ${b})`;
}

export async function rotateImageFile(
  file: File,
  degrees: number
): Promise<File> {
  const normalized = ((degrees % 360) + 360) % 360;
  if (normalized === 0) return file;

  const img = await loadImage(file);
  const rad = (normalized * Math.PI) / 180;
  const sin = Math.abs(Math.sin(rad));
  const cos = Math.abs(Math.cos(rad));
  const w = img.width;
  const h = img.height;
  const newW = Math.round(w * cos + h * sin);
  const newH = Math.round(w * sin + h * cos);

  const canvas = document.createElement("canvas");
  canvas.width = newW;
  canvas.height = newH;
  const ctx = canvas.getContext("2d")!;

  ctx.fillStyle = sampleBackgroundColor(ctx, img);
  ctx.fillRect(0, 0, newW, newH);

  ctx.translate(newW / 2, newH / 2);
  ctx.rotate(rad);
  ctx.drawImage(img, -w / 2, -h / 2);

  const isPng = file.type === "image/png";
  const mime = isPng ? "image/png" : "image/jpeg";
  const blob: Blob = await new Promise((res, rej) => {
    canvas.toBlob(
      (b) => (b ? res(b) : rej(new Error("toBlob failed"))),
      mime,
      0.95
    );
  });

  const baseName = file.name.replace(/\.[^.]+$/, "");
  const ext = isPng ? "png" : "jpg";
  return new File([blob], `${baseName}_rot${normalized}.${ext}`, { type: mime });
}
