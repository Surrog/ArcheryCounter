import React, { useState } from 'react';
import {
  ActivityIndicator,
  Image,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { RingOverlay, DEFAULT_VISIBILITY, OverlayVisibility } from '../components/RingOverlay';
import { useArcheryScorer } from '../useArcheryScorer';
import { pointInPolygon } from '../targetDetection';

type VisKey = keyof OverlayVisibility;
const TOGGLE_LABELS: { key: VisKey; label: string }[] = [
  { key: 'rings',    label: 'Rings'    },
  { key: 'rays',     label: 'Rays'     },
  { key: 'boundary', label: 'Boundary' },
  { key: 'arrows',   label: 'Arrows'   },
];

export function HomeScreen() {
  const { imageUri, targets, arrows, ringPoints, imageWidth, imageHeight, loading, error, pickAndProcess, reset } =
    useArcheryScorer();
  const [visibility, setVisibility] = useState<OverlayVisibility>(DEFAULT_VISIBILITY);
  const [selectedTargetIdx, setSelectedTargetIdx] = useState<number | null>(null);

  const toggleLayer = (key: VisKey) =>
    setVisibility(v => ({ ...v, [key]: !v[key] }));

  const hasResult = imageUri && targets && imageWidth && imageHeight;
  const multiTarget = (targets?.length ?? 0) > 1;

  // Filter arrows to selected target when one is selected.
  const visibleArrows = selectedTargetIdx !== null && targets
    ? (arrows ?? []).filter(a =>
        pointInPolygon({ x: a.tip[0], y: a.tip[1] }, { points: targets[selectedTargetIdx].paperBoundary }),
      )
    : arrows;

  const totalRings = targets ? targets.reduce((s, t) => s + t.rings.length, 0) : 0;

  return (
    <SafeAreaView style={styles.safe}>
      <Text style={styles.title}>Archery Counter</Text>

      {/* ── Image area ───────────────────────────────────────────── */}
      <View style={styles.imageContainer}>
        {hasResult ? (
          <>
            <Image source={{ uri: imageUri }} style={styles.image} resizeMode="contain" />
            <RingOverlay
              targets={targets}
              selectedTargetIdx={selectedTargetIdx}
              arrows={visibleArrows}
              ringPoints={ringPoints}
              imageNaturalWidth={imageWidth}
              imageNaturalHeight={imageHeight}
              visibility={visibility}
            />
          </>
        ) : (
          <View style={styles.placeholder}>
            <Text style={styles.placeholderText}>
              {loading ? 'Detecting…' : 'No image selected'}
            </Text>
          </View>
        )}

        {loading && (
          <View style={styles.loadingOverlay}>
            <ActivityIndicator size="large" color="#FFFFFF" />
          </View>
        )}
      </View>

      {/* ── Target selector (multi-target only) ───────────────────── */}
      {hasResult && multiTarget && (
        <View style={styles.toggleRow}>
          <Pressable
            style={[styles.toggleButton, selectedTargetIdx === null && styles.toggleButtonOn]}
            onPress={() => setSelectedTargetIdx(null)}>
            <Text style={[styles.toggleText, selectedTargetIdx === null && styles.toggleTextOn]}>
              All
            </Text>
          </Pressable>
          {targets.map((_, i) => (
            <Pressable
              key={i}
              style={[styles.toggleButton, selectedTargetIdx === i && styles.toggleButtonOn]}
              onPress={() => setSelectedTargetIdx(selectedTargetIdx === i ? null : i)}>
              <Text style={[styles.toggleText, selectedTargetIdx === i && styles.toggleTextOn]}>
                {`T${i + 1}`}
              </Text>
            </Pressable>
          ))}
        </View>
      )}

      {/* ── Layer toggles ─────────────────────────────────────────── */}
      {hasResult && (
        <View style={styles.toggleRow}>
          {TOGGLE_LABELS.map(({ key, label }) => (
            <Pressable
              key={key}
              style={[styles.toggleButton, visibility[key] && styles.toggleButtonOn]}
              onPress={() => toggleLayer(key)}>
              <Text style={[styles.toggleText, visibility[key] && styles.toggleTextOn]}>
                {label}
              </Text>
            </Pressable>
          ))}
        </View>
      )}

      {/* ── Stats badge ───────────────────────────────────────────── */}
      {hasResult && (
        <Text style={styles.statsText}>
          {targets.length} target{targets.length !== 1 ? 's' : ''} · {totalRings} rings · {(visibleArrows?.length ?? 0)} arrow{(visibleArrows?.length ?? 0) !== 1 ? 's' : ''} detected
        </Text>
      )}

      {/* ── Error ────────────────────────────────────────────────── */}
      {error && <Text style={styles.errorText}>{error}</Text>}

      {/* ── Actions ──────────────────────────────────────────────── */}
      <View style={styles.actions}>
        <Pressable
          style={[styles.button, loading && styles.buttonDisabled]}
          onPress={pickAndProcess}
          disabled={loading}>
          <Text style={styles.buttonText}>
            {hasResult ? 'Pick another photo' : 'Pick photo from library'}
          </Text>
        </Pressable>

        {hasResult && (
          <Pressable style={[styles.button, styles.buttonSecondary]} onPress={() => { setSelectedTargetIdx(null); reset(); }}>
            <Text style={styles.buttonText}>Clear</Text>
          </Pressable>
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: '#111',
  },
  title: {
    fontSize: 22,
    fontWeight: '700',
    color: '#FFF',
    textAlign: 'center',
    paddingVertical: 12,
  },
  imageContainer: {
    flex: 1,
    margin: 12,
    borderRadius: 8,
    overflow: 'hidden',
    backgroundColor: '#222',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  placeholder: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  placeholderText: {
    color: '#666',
    fontSize: 16,
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFill,
    backgroundColor: 'rgba(0,0,0,0.5)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  toggleRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 6,
    paddingHorizontal: 16,
  },
  toggleButton: {
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#555',
    paddingHorizontal: 10,
    paddingVertical: 5,
    backgroundColor: '#222',
  },
  toggleButtonOn: {
    backgroundColor: '#2266CC',
    borderColor: '#2266CC',
  },
  toggleText: {
    color: '#888',
    fontSize: 13,
    fontWeight: '500',
  },
  toggleTextOn: {
    color: '#FFF',
  },
  statsText: {
    color: '#AAA',
    textAlign: 'center',
    marginBottom: 4,
    fontSize: 14,
  },
  errorText: {
    color: '#FF6666',
    textAlign: 'center',
    paddingHorizontal: 16,
    marginBottom: 8,
    fontSize: 13,
  },
  actions: {
    paddingHorizontal: 16,
    paddingBottom: 8,
    gap: 8,
  },
  button: {
    backgroundColor: '#2266CC',
    borderRadius: 8,
    paddingVertical: 14,
    alignItems: 'center',
  },
  buttonSecondary: {
    backgroundColor: '#444',
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  buttonText: {
    color: '#FFF',
    fontWeight: '600',
    fontSize: 15,
  },
});
