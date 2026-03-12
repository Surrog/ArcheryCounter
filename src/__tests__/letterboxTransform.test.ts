import { computeLetterboxTransform } from '../letterboxTransform';

describe('computeLetterboxTransform', () => {
  describe('wider image than view (letterbox: bars top/bottom)', () => {
    // Image 1000x500 (aspect 2:1) inside view 800x600 (aspect 4:3)
    const t = computeLetterboxTransform(1000, 500, 800, 600);

    it('fills the full view width', () => {
      // renderedWidth = 800
      expect(t.scale).toBeCloseTo(800 / 1000);
    });

    it('has no horizontal offset', () => {
      expect(t.offsetX).toBe(0);
    });

    it('centers vertically', () => {
      // renderedHeight = 800/2 = 400; offsetY = (600-400)/2 = 100
      expect(t.offsetY).toBeCloseTo(100);
    });
  });

  describe('taller image than view (pillarbox: bars left/right)', () => {
    // Image 500x1000 (aspect 1:2) inside view 800x600 (aspect 4:3)
    const t = computeLetterboxTransform(500, 1000, 800, 600);

    it('fills the full view height', () => {
      // renderedHeight = 600; renderedWidth = 600*0.5 = 300; scale = 300/500
      expect(t.scale).toBeCloseTo(300 / 500);
    });

    it('centers horizontally', () => {
      // offsetX = (800-300)/2 = 250
      expect(t.offsetX).toBeCloseTo(250);
    });

    it('has no vertical offset', () => {
      expect(t.offsetY).toBe(0);
    });
  });

  describe('image aspect matches view exactly', () => {
    // Image 800x600 in view 800x600 — no bars
    const t = computeLetterboxTransform(800, 600, 800, 600);

    it('scale is 1', () => {
      expect(t.scale).toBeCloseTo(1);
    });

    it('no offsets', () => {
      expect(t.offsetX).toBeCloseTo(0);
      expect(t.offsetY).toBeCloseTo(0);
    });
  });

  it('maps image center to view center', () => {
    const iw = 1000, ih = 500, vw = 800, vh = 600;
    const t = computeLetterboxTransform(iw, ih, vw, vh);
    const cx = (iw / 2) * t.scale + t.offsetX;
    const cy = (ih / 2) * t.scale + t.offsetY;
    expect(cx).toBeCloseTo(vw / 2);
    expect(cy).toBeCloseTo(vh / 2);
  });
});
