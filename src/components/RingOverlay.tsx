import React, { useMemo, useState } from 'react';
import { LayoutChangeEvent, StyleSheet, View } from 'react-native';
import Svg, { Circle, Line, Path, Polygon } from 'react-native-svg';
import type { SplineRing } from '../spline';
import { sampleClosedSpline } from '../spline';
import type { TargetBoundary, Pixel } from '../targetDetection';
import type { ScoredArrow } from '../scoring';
import { computeLetterboxTransform } from '../letterboxTransform';

export interface OverlayVisibility {
  rings: boolean;
  rays: boolean;
  boundary: boolean;
  arrows: boolean;
}

export const DEFAULT_VISIBILITY: OverlayVisibility = {
  rings: true,
  rays: false,
  boundary: true,
  arrows: true,
};

interface Props {
  rings: SplineRing[];
  /** Detected target paper boundary polygon in image pixels */
  paperBoundary?: TargetBoundary | null;
  /** Detected arrows */
  arrows?: ScoredArrow[] | null;
  /** Raw per-ray ring transition points (index 0 = innermost ring) */
  ringPoints?: Pixel[][] | null;
  /** Original pixel dimensions of the source image */
  imageNaturalWidth: number;
  imageNaturalHeight: number;
  /** Which overlay layers to show */
  visibility?: OverlayVisibility;
}

// Ring stroke colours, index 0 = innermost (bullseye), index 9 = outermost.
// Approximate the standard WA/FITA target face colours.
const RING_STROKE: string[] = [
  '#FFB800', // 10 — yellow inner (gold / bullseye)
  '#FFDD00', // 9 — yellow outer
  '#CC0000', // 8 — red inner
  '#FF4444', // 7 — red outer
  '#0000CC', // 6 — blue inner
  '#4444FF', // 5 — blue outer
  '#111111', // 4 — black inner
  '#333333', // 3 — black outer
  '#FFFFFF', // 2 — white inner
  '#AAAAAA', // 1 — white outer
];

interface ViewSize {
  width: number;
  height: number;
}

/**
 * Transparent SVG layer that sits on top of an <Image resizeMode="contain">.
 * It compensates for the letterboxing that "contain" introduces so that
 * the ring paths are aligned with the actual pixels of the displayed photo.
 */
export function RingOverlay({
  rings,
  paperBoundary,
  arrows,
  ringPoints,
  imageNaturalWidth,
  imageNaturalHeight,
  visibility = DEFAULT_VISIBILITY,
}: Props) {
  const [viewSize, setViewSize] = useState<ViewSize>({ width: 0, height: 0 });

  const transform = useMemo(() => {
    if (!viewSize.width || !viewSize.height) return null;
    return computeLetterboxTransform(imageNaturalWidth, imageNaturalHeight, viewSize.width, viewSize.height);
  }, [viewSize, imageNaturalWidth, imageNaturalHeight]);

  const handleLayout = (e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    setViewSize({ width, height });
  };

  const tx = (px: number) => px * (transform?.scale ?? 1) + (transform?.offsetX ?? 0);
  const ty = (py: number) => py * (transform?.scale ?? 1) + (transform?.offsetY ?? 0);

  return (
    <View style={StyleSheet.absoluteFill} onLayout={handleLayout} pointerEvents="none">
      {transform && viewSize.width > 0 && (
        <Svg width={viewSize.width} height={viewSize.height}>

          {/* Paper boundary — dashed lime quadrilateral */}
          {visibility.boundary && paperBoundary && paperBoundary.points.length >= 3 && (() => {
            const points = paperBoundary.points
              .map(([px, py]) => `${tx(px)},${ty(py)}`)
              .join(' ');
            return (
              <Polygon
                points={points}
                fill="none"
                stroke="#00FF88"
                strokeWidth={2}
                strokeDasharray="8 4"
              />
            );
          })()}

          {/* Rays — lines from target centre to each detected ring transition point */}
          {visibility.rays && ringPoints && ringPoints.length > 0 && (() => {
            // Compute centre from innermost ring
            const inner = rings[0];
            if (!inner) return null;
            const icx = inner.points.reduce((s, p) => s + p[0], 0) / inner.points.length;
            const icy = inner.points.reduce((s, p) => s + p[1], 0) / inner.points.length;
            // Collect all unique ray endpoints from the outermost detected ring
            const outerPts = ringPoints[ringPoints.length - 1] ?? [];
            return outerPts.map((pt, i) => (
              <Line
                key={i}
                x1={tx(icx)} y1={ty(icy)}
                x2={tx(pt.x)} y2={ty(pt.y)}
                stroke="rgba(255,255,255,0.25)"
                strokeWidth={0.8}
              />
            ));
          })()}

          {/* Scoring rings */}
          {visibility.rings && rings.map((ring, i) => {
            const sampled = sampleClosedSpline(ring.points, 120);
            if (sampled.length < 2) return null;
            const [first, ...rest] = sampled;
            let d = `M ${tx(first[0])} ${ty(first[1])}`;
            for (const [px, py] of rest) {
              d += ` L ${tx(px)} ${ty(py)}`;
            }
            d += ' Z';
            return (
              <Path
                key={i}
                d={d}
                fill="none"
                stroke={RING_STROKE[i] ?? '#00FF00'}
                strokeWidth={1.5}
              />
            );
          })}

          {/* Arrows — tip dot */}
          {visibility.arrows && arrows && arrows.map((arrow, i) => {
            const [tipX, tipY] = arrow.tip;
            return (
              <Circle
                key={i}
                cx={tx(tipX)} cy={ty(tipY)}
                r={5}
                fill="#FF6600"
                stroke="#FFFFFF"
                strokeWidth={1}
              />
            );
          })}

        </Svg>
      )}
    </View>
  );
}
