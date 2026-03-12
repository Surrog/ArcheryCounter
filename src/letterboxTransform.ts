export interface LetterboxTransform {
  scale: number;
  offsetX: number;
  offsetY: number;
}

/**
 * Computes the scale and offset needed to map image-pixel coordinates onto a
 * view that displays the image with `resizeMode="contain"` (letterbox/pillarbox).
 */
export function computeLetterboxTransform(
  imageNaturalWidth: number,
  imageNaturalHeight: number,
  viewWidth: number,
  viewHeight: number,
): LetterboxTransform {
  const imageAspect = imageNaturalWidth / imageNaturalHeight;
  const viewAspect = viewWidth / viewHeight;

  let renderedWidth: number;
  let renderedHeight: number;
  let offsetX: number;
  let offsetY: number;

  if (imageAspect > viewAspect) {
    // Image wider than view — letterbox top and bottom
    renderedWidth = viewWidth;
    renderedHeight = viewWidth / imageAspect;
    offsetX = 0;
    offsetY = (viewHeight - renderedHeight) / 2;
  } else {
    // Image taller — pillarbox left and right
    renderedHeight = viewHeight;
    renderedWidth = viewHeight * imageAspect;
    offsetX = (viewWidth - renderedWidth) / 2;
    offsetY = 0;
  }

  return {
    scale: renderedWidth / imageNaturalWidth,
    offsetX,
    offsetY,
  };
}
