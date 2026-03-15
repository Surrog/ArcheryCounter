import React, { useMemo, useState } from 'react';
import { LayoutChangeEvent, StyleSheet, View } from 'react-native';
import Svg, { Ellipse, G, Polygon } from 'react-native-svg';
import type { RingEllipse } from '../NativeArcheryCounter';
import type { TargetBoundary } from '../targetDetection';
import { computeLetterboxTransform } from '../letterboxTransform';

interface Props {
  rings: RingEllipse[];
  /** Detected target paper boundary polygon in image pixels */
  paperBoundary?: TargetBoundary | null;
  /** Original pixel dimensions of the source image */
  imageNaturalWidth: number;
  imageNaturalHeight: number;
}

// Ring stroke colours, outermost (index 0) to innermost (index 9).
// Approximate the standard WA/FITA target face colours.
const RING_STROKE: string[] = [
  '#AAAAAA', // 1 — white outer
  '#FFFFFF', // 2 — white inner
  '#333333', // 3 — black outer
  '#111111', // 4 — black inner
  '#4444FF', // 5 — blue outer
  '#0000CC', // 6 — blue inner
  '#FF4444', // 7 — red outer
  '#CC0000', // 8 — red inner
  '#FFDD00', // 9 — yellow outer
  '#FFB800', // 10 — yellow inner (gold / bullseye)
];

interface ViewSize {
  width: number;
  height: number;
}

/**
 * Transparent SVG layer that sits on top of an <Image resizeMode="contain">.
 * It compensates for the letterboxing that "contain" introduces so that
 * the ellipses are aligned with the actual pixels of the displayed photo.
 */
export function RingOverlay({ rings, paperBoundary, imageNaturalWidth, imageNaturalHeight }: Props) {
  const [viewSize, setViewSize] = useState<ViewSize>({ width: 0, height: 0 });

  const transform = useMemo(() => {
    if (!viewSize.width || !viewSize.height) return null;
    return computeLetterboxTransform(imageNaturalWidth, imageNaturalHeight, viewSize.width, viewSize.height);
  }, [viewSize, imageNaturalWidth, imageNaturalHeight]);

  const handleLayout = (e: LayoutChangeEvent) => {
    const { width, height } = e.nativeEvent.layout;
    setViewSize({ width, height });
  };

  return (
    <View style={StyleSheet.absoluteFill} onLayout={handleLayout} pointerEvents="none">
      {transform && viewSize.width > 0 && (
        <Svg width={viewSize.width} height={viewSize.height}>
          {/* Target paper boundary — dashed lime quadrilateral */}
          {paperBoundary && paperBoundary.points.length >= 3 && (() => {
            const points = paperBoundary.points
              .map(([px, py]) => `${px * transform.scale + transform.offsetX},${py * transform.scale + transform.offsetY}`)
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

          {/* Scoring rings */}
          {rings.map((ring, i) => {
            const cx = ring.centerX * transform.scale + transform.offsetX;
            const cy = ring.centerY * transform.scale + transform.offsetY;
            const rx = (ring.width / 2) * transform.scale;
            const ry = (ring.height / 2) * transform.scale;

            return (
              // <G rotation> with origin rotates the ellipse around its own center,
              // replicating the cv::RotatedRect angle convention.
              <G key={i} rotation={ring.angle} origin={`${cx}, ${cy}`}>
                <Ellipse
                  cx={cx}
                  cy={cy}
                  rx={rx}
                  ry={ry}
                  fill="none"
                  stroke={RING_STROKE[i] ?? '#00FF00'}
                  strokeWidth={1.5}
                />
              </G>
            );
          })}
        </Svg>
      )}
    </View>
  );
}
