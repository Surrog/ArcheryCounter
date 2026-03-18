import React from 'react';
import {
  ActivityIndicator,
  Image,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { RingOverlay } from '../components/RingOverlay';
import { useArcheryScorer } from '../useArcheryScorer';

export function HomeScreen() {
  const { imageUri, rings, paperBoundary, imageWidth, imageHeight, loading, error, pickAndProcess, reset } =
    useArcheryScorer();

  const hasResult = imageUri && rings && imageWidth && imageHeight;

  return (
    <SafeAreaView style={styles.safe}>
      <Text style={styles.title}>Archery Counter</Text>

      {/* ── Image area ───────────────────────────────────────────── */}
      <View style={styles.imageContainer}>
        {hasResult ? (
          <>
            <Image source={{ uri: imageUri }} style={styles.image} resizeMode="contain" />
            {/* RingOverlay sits on top and compensates for contain letterboxing */}
            <RingOverlay
              rings={rings}
              paperBoundary={paperBoundary}
              imageNaturalWidth={imageWidth}
              imageNaturalHeight={imageHeight}
            />
          </>
        ) : (
          <View style={styles.placeholder}>
            <Text style={styles.placeholderText}>
              {loading ? 'Detecting rings…' : 'No image selected'}
            </Text>
          </View>
        )}

        {loading && (
          <View style={styles.loadingOverlay}>
            <ActivityIndicator size="large" color="#FFFFFF" />
          </View>
        )}
      </View>

      {/* ── Ring count badge ─────────────────────────────────────── */}
      {hasResult && (
        <Text style={styles.ringCount}>
          {rings.length} ring{rings.length !== 1 ? 's' : ''} detected
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
          <Pressable style={[styles.button, styles.buttonSecondary]} onPress={reset}>
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
  ringCount: {
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
