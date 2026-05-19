import React, { useState, useRef, useEffect, useCallback, ChangeEvent } from 'react';
import { 
  Upload, 
  Settings2, 
  Play, 
  Pause, 
  RefreshCcw, 
  Download, 
  Trash2, 
  Image as ImageIcon,
  Palette,
  Maximize2,
  Volume2,
  VolumeX,
  Eye,
  EyeOff,
  Cpu,
  Sun,
  Sparkles,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import * as bodySegmentation from '@tensorflow-models/body-segmentation';
import '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-converter';

// Utility for truncation
const truncate = (str: string, n: number) => {
  return str.length > n ? str.substr(0, n - 1) + "..." : str;
};

// Chroma Key Component
export default function App() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [tolerance, setTolerance] = useState(70);
  const [smoothing, setSmoothing] = useState(20);
  const [targetColor, setTargetColor] = useState({ r: 0, g: 255, b: 0 }); // Default Green
  const [bgMode, setBgMode] = useState<'transparent' | 'color' | 'image'>('transparent');
  const [bgColor, setBgColor] = useState('#000000');
  const [bgImage, setBgImage] = useState<string | null>(null);
  const [showMask, setShowMask] = useState(false);
  const [lightWrap, setLightWrap] = useState(30);
  const [matteShrink, setMatteShrink] = useState(1);
  const [matteFeather, setMatteFeather] = useState(10);
  const [despillStrength, setDespillStrength] = useState(80);
  const [colorMatch, setColorMatch] = useState(true);
  const [relighting, setRelighting] = useState(true);
  const [useAISegmentation, setUseAISegmentation] = useState(false);
  const [aiFrequency, setAiFrequency] = useState(3); // Process AI every N frames
  const [processingScale, setProcessingScale] = useState(0.75); // 0.5 to 1.0
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);

  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const bgImgRef = useRef<HTMLImageElement | null>(null);
  const bgCanvasCacheRef = useRef<HTMLCanvasElement | null>(null);
  const bgStatsRef = useRef<{ 
    avgR: number, 
    avgG: number, 
    avgB: number,
    lightDir: { x: number, y: number } 
  }>({ avgR: 0, avgG: 0, avgB: 0, lightDir: { x: 0, y: 0 } });
  
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const aiMaskCacheRef = useRef<Uint8Array | null>(null);
  const frameCountRef = useRef<number>(0);
  const segmenterRef = useRef<bodySegmentation.BodySegmenter | null>(null);
  const processingRef = useRef<boolean>(false);
  const requestRef = useRef<number | null>(null);

  useEffect(() => {
    document.title = "Green screen remover | Professional Background Removal";
  }, []);

  useEffect(() => {
    if (bgImage) {
      const img = new Image();
      img.src = bgImage;
      img.onload = () => {
        bgImgRef.current = img;
        // Pre-render background image to cache if video dimensions are known
        if (videoRef.current && videoRef.current.videoWidth > 0) {
          updateBgCache();
        }
      };
    } else {
      bgImgRef.current = null;
      bgCanvasCacheRef.current = null;
    }
  }, [bgImage]);

  const updateBgCache = useCallback(() => {
    const video = videoRef.current;
    const img = bgImgRef.current;
    if (!video || !img || video.videoWidth === 0) return;

    if (!bgCanvasCacheRef.current) {
      bgCanvasCacheRef.current = document.createElement('canvas');
    }
    const cache = bgCanvasCacheRef.current;
    cache.width = video.videoWidth;
    cache.height = video.videoHeight;
    const ctx = cache.getContext('2d');
    if (ctx) {
      // Cover logic for background image
      const vRatio = cache.width / cache.height;
      const iRatio = img.width / img.height;
      let sx, sy, sw, sh;
      if (iRatio > vRatio) {
        sh = img.height;
        sw = img.height * vRatio;
        sx = (img.width - sw) / 2;
        sy = 0;
      } else {
        sw = img.width;
        sh = img.width / vRatio;
        sx = 0;
        sy = (img.height - sh) / 2;
      }
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, cache.width, cache.height);
      
      // Analyze background statistics
      const bgData = ctx.getImageData(0, 0, cache.width, cache.height).data;
      let totalR = 0, totalG = 0, totalB = 0;
      let leftLum = 0, rightLum = 0, topLum = 0, bottomLum = 0;
      
      const step = 10; // Sample every 10th pixel for speed
      let count = 0;
      for (let i = 0; i < bgData.length; i += 4 * step) {
        const r = bgData[i];
        const g = bgData[i + 1];
        const b = bgData[i + 2];
        const lum = (0.2126 * r + 0.7152 * g + 0.0722 * b);
        
        totalR += r; totalG += g; totalB += b;
        
        const px = (i / 4) % cache.width;
        const py = Math.floor((i / 4) / cache.width);
        
        if (px < cache.width / 3) leftLum += lum;
        if (px > (cache.width * 2) / 3) rightLum += lum;
        if (py < cache.height / 3) topLum += lum;
        if (py > (cache.height * 2) / 3) bottomLum += lum;
        
        count++;
      }
      
      bgStatsRef.current = {
        avgR: totalR / count,
        avgG: totalG / count,
        avgB: totalB / count,
        lightDir: {
          x: (rightLum - leftLum) / (leftLum + rightLum + 0.1),
          y: (bottomLum - topLum) / (topLum + bottomLum + 0.1)
        }
      };
    }
  }, []);

  useEffect(() => {
    if (videoRef.current) {
      videoRef.current.volume = volume;
      videoRef.current.muted = isMuted;
    }
  }, [volume, isMuted]);

  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  const toggleFullscreen = () => {
    const container = canvasRef.current?.parentElement;
    if (container) {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else {
        container.requestFullscreen().catch(console.error);
      }
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setVideoFile(file);
      const url = URL.createObjectURL(file);
      setVideoUrl(url);
      setIsMuted(false); // Enable sound by default when new video is loaded
    }
  };

  const handleBgImageChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const url = URL.createObjectURL(e.target.files[0]);
      setBgImage(url);
    }
  };

  useEffect(() => {
    const loadModel = async () => {
      setIsModelLoading(true);
      try {
        const model = bodySegmentation.SupportedModels.MediaPipeSelfieSegmentation;
        const segmenterConfig = {
          runtime: 'tfjs',
          modelType: 'general',
        } as const;
        segmenterRef.current = await bodySegmentation.createSegmenter(model, segmenterConfig);
      } catch (err) {
        console.error("Failed to load segmenter:", err);
      } finally {
        setIsModelLoading(false);
      }
    };
    if (useAISegmentation && !segmenterRef.current) {
      loadModel();
    }
  }, [useAISegmentation]);

  const processFrame = useCallback(async () => {
    if (processingRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    processingRef.current = true;
    frameCountRef.current++;

    try {
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) return;

      // Quality optimization: process at lower resolution if quality < 1
      const procWidth = Math.floor(video.videoWidth * processingScale);
      const procHeight = Math.floor(video.videoHeight * processingScale);
      
      // We still want the main canvas to match video resolution for export
      // But we can use an offscreen canvas for the heavy pixel manipulation
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      // Draw video to a smaller internal surface if scaled
      let frame;
      if (processingScale < 1.0) {
        const offCtx = new OffscreenCanvas(procWidth, procHeight).getContext('2d', { willReadFrequently: true });
        if (!offCtx) throw new Error("Could not get offscreen context");
        offCtx.drawImage(video, 0, 0, procWidth, procHeight);
        frame = offCtx.getImageData(0, 0, procWidth, procHeight);
      } else {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
      }
      
      const data = frame.data;

      // Optional: Get AI segmentation mask with throttling
      let aiMask: Uint8Array | null = aiMaskCacheRef.current;
      if (useAISegmentation && segmenterRef.current && (frameCountRef.current % aiFrequency === 0 || !aiMask)) {
        try {
          const segmentation = await segmenterRef.current.segmentPeople(video);
          const mask = await segmentation[0].mask.toUint8Array();
          
          // If we are processing at a different scale, we might need to handle mask index mapping
          // But for now, we just cache it. 
          // Note: mask.toUint8Array usually returns a mask at the same size as the input video
          aiMask = mask;
          aiMaskCacheRef.current = mask;
        } catch (err) {
          console.error("Segmentation error:", err);
        }
      }

      // Prepare background color for light wrap
      let bgr = 0, bgg = 0, bgb = 0;
      if (bgMode === 'color') {
        const hex = bgColor.replace('#', '');
        bgr = parseInt(hex.substring(0, 2), 16);
        bgg = parseInt(hex.substring(2, 4), 16);
        bgb = parseInt(hex.substring(4, 6), 16);
      }

      // Chroma target colors
      const tr = targetColor.r;
      const tg = targetColor.g;
      const tb = targetColor.b;

      const isGreen = tg > tr && tg > tb;
      const isBlue = tb > tr && tb > tg;
      const isRed = tr > tg && tr > tb;

      const intensityFactor = lightWrap / 100;
      const videoWidthValue = video.videoWidth;
      const videoHeightValue = video.videoHeight;

      for (let i = 0; i < data.length; i += 4) {
        let r = data[i];
        let g = data[i + 1];
        let b = data[i + 2];

        // LUMINANCE CALCULATION (REC.709)
        const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;

        // Calculate key score
        let score = 0;
        if (isGreen) {
          // Optimized for hair: use a higher bias for blue/red separation
          score = (g - Math.max(r, b) * 1.05) * 2.0;
        } else if (isBlue) {
          score = (b - Math.max(r, g) * 1.05) * 2.0;
        } else if (isRed) {
          score = (r - Math.max(g, b) * 1.05) * 2.0;
        } else {
          const dist = Math.sqrt(Math.pow(r - tr, 2) + Math.pow(g - tg, 2) + Math.pow(b - tb, 2));
          score = 255 - dist;
        }

        // Map score to alpha
        let alpha = 255;
        const shrinkOffset = (matteShrink * 2); 
        const lowBound = tolerance + shrinkOffset;
        const totalSmoothing = smoothing + matteFeather;
        const highBound = lowBound + totalSmoothing + 0.1;

        if (score > highBound) {
          alpha = 0;
        } else if (score > lowBound) {
          const ratio = (score - lowBound) / (highBound - lowBound);
          // Smoother Step: 6x^5 - 15x^4 + 10x^3
          const smoothRatio = ratio * ratio * ratio * (ratio * (ratio * 6 - 15) + 10);
          alpha = 255 * (1 - smoothRatio);

          // Transparency/Glass Protection: If it's bright and somewhat green, 
          // it might be a reflection. We reduce the "killing" force of the alpha.
          if (lum > 180 && isGreen && ratio < 0.8) {
             alpha = Math.max(alpha, (lum - 150) * 0.5);
          }
        }

        // --- SUBJECT RECOGNITION REFINEMENT ---
        if (aiMask) {
          // Map processing pixel back to AI mask coordinate
          let maskIndex;
          if (processingScale < 1.0) {
            const px = (i / 4) % procWidth;
            const py = Math.floor((i / 4) / procWidth);
            const mx = Math.floor(px / processingScale);
            const my = Math.floor(py / processingScale);
            maskIndex = my * videoWidthValue + mx;
          } else {
            maskIndex = i / 4;
          }
          
          if (maskIndex < aiMask.length) {
            const aiConf = aiMask[maskIndex] / 255;
            if (aiConf > 0.5) {
              const protection = (aiConf - 0.5) * 2;
              alpha = Math.max(alpha, 255 * protection * 0.8);
            } else if (aiConf < 0.2) {
              const killFactor = (0.2 - aiConf) * 5;
              if (score > tolerance - 20) {
                alpha *= (1 - killFactor);
              }
            }
          }
        }

        // ... shadow preservation and spill suppression ...

      // --- SHADOW PRESERVATION ---
      if (isGreen && alpha < 200) {
        const luminance = (r * 0.299 + g * 0.587 + b * 0.114);
        if (luminance < 120) {
          const shadowFactor = Math.max(0, 1 - (luminance / 120));
          const keyIntensity = Math.min(1.0, score / 120);
          const recovery = shadowFactor * (1 - keyIntensity) * 0.7; 
          alpha = Math.min(255, alpha + (recovery * 255));
          const avg = (r + b) / 2;
          const neutralizationStrength = shadowFactor * 0.9;
          g = g * (1 - neutralizationStrength) + avg * neutralizationStrength;
        }
      }

        // --- ADVANCED SPILL SUPPRESSION (Luminance Preserving) ---
        if (alpha > 0) {
          const factor = despillStrength / 100;
          if (isGreen) {
            const avgOther = (r + b) / 2;
            if (g > avgOther) {
              const targetG = avgOther;
              g = g * (1 - factor) + targetG * factor;
            }
          } else if (isBlue) {
            const avgOther = (r + g) / 2;
            if (b > avgOther) {
              const targetB = avgOther;
              b = b * (1 - factor) + targetB * factor;
            }
          } else if (isRed) {
            const avgOther = (g + b) / 2;
            if (r > avgOther) {
              const targetR = avgOther;
              r = r * (1 - factor) + targetR * factor;
            }
          }
  
          // --- COLOR HARMONIZATION ---
          if (bgMode === 'image' && colorMatch) {
            // Apply a subtle tint towards the background's average color
            const matchStrength = 0.15; // 15% blend
            const { avgR, avgG, avgB } = bgStatsRef.current;
            r = r * (1 - matchStrength) + avgR * matchStrength;
            g = g * (1 - matchStrength) + avgG * matchStrength;
            b = b * (1 - matchStrength) + avgB * matchStrength;
          }
  
          // --- DIRECTIONAL RELIGHTING ---
          if (bgMode === 'image' && relighting) {
            const { x: dx, y: dy } = bgStatsRef.current.lightDir;
            const px = ((i / 4) % procWidth) / procWidth - 0.5; // -0.5 to 0.5
            const py = (Math.floor((i / 4) / procWidth) / procHeight) - 0.5;
            
            // Interaction between pixel position and background light direction
            const lightInfluence = (px * dx + py * dy) * 40; // Max 20 unit adjustment
            r = Math.min(255, Math.max(0, r + lightInfluence));
            g = Math.min(255, Math.max(0, g + lightInfluence));
            b = Math.min(255, Math.max(0, b + lightInfluence));
          }
  
          // --- LIGHT WRAPPING ---
          if (intensityFactor > 0 && alpha > 0 && alpha < 250) {
            const edgeWeight = (1.0 - alpha / 255) * intensityFactor;
            let targetBGR = bgr, targetBGG = bgg, targetBGB = bgb;
            
            if (bgMode === 'image') {
              targetBGR = bgStatsRef.current.avgR;
              targetBGG = bgStatsRef.current.avgG;
              targetBGB = bgStatsRef.current.avgB;
            }
            
            r = r * (1 - edgeWeight) + targetBGR * edgeWeight;
            g = g * (1 - edgeWeight) + targetBGG * edgeWeight;
            b = b * (1 - edgeWeight) + targetBGB * edgeWeight;
          }
        }

      if (showMask) {
        data[i] = alpha;
        data[i + 1] = alpha;
        data[i + 2] = alpha;
        data[i + 3] = 255;
      } else {
        data[i] = r;
        data[i + 1] = g;
        data[i + 2] = b;
        data[i + 3] = alpha;
      }
    }

    if (processingScale < 1.0) {
      // Scale back up to main canvas
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = procWidth;
      tempCanvas.height = procHeight;
      tempCanvas.getContext('2d')?.putImageData(frame, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.putImageData(frame, 0, 0);
    }

    // Composite background
    if (bgMode !== 'transparent') {
      ctx.globalCompositeOperation = 'destination-over';
      if (bgMode === 'color') {
        ctx.fillStyle = bgColor;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
      } else if (bgMode === 'image' && bgImgRef.current) {
        // Use cached background canvas if possible
        if (bgCanvasCacheRef.current) {
          ctx.drawImage(bgCanvasCacheRef.current, 0, 0, canvas.width, canvas.height);
        } else {
          // Fallback but try to update cache
          ctx.drawImage(bgImgRef.current, 0, 0, canvas.width, canvas.height);
          updateBgCache();
        }
      }
      ctx.globalCompositeOperation = 'source-over';
    }

    if (!video.paused && !video.ended) {
      requestRef.current = requestAnimationFrame(processFrame);
    }
    } catch (err) {
      console.error("Frame processing error:", err);
    } finally {
      processingRef.current = false;
    }
  }, [targetColor, tolerance, smoothing, bgMode, bgColor, showMask, lightWrap, useAISegmentation, matteShrink, matteFeather, processingScale, aiFrequency, updateBgCache, despillStrength, colorMatch, relighting]);

  useEffect(() => {
    if (isPlaying) {
      requestRef.current = requestAnimationFrame(processFrame);
    } else {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      // Process one frame even if paused to see adjustments
      processFrame();
    }
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [isPlaying, processFrame]);

  // Clean up Object URLs
  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
      if (bgImage) URL.revokeObjectURL(bgImage);
    };
  }, [videoUrl, bgImage]);

  const togglePlay = () => {
    if (videoRef.current) {
      if (videoRef.current.paused) {
        videoRef.current.play().then(() => setIsPlaying(true)).catch(console.error);
      } else {
        videoRef.current.pause();
        setIsPlaying(false);
      }
    }
  };

  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const audioSourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const wasCancelledRef = useRef(false);

  const startExport = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || isExporting) return;

    try {
      wasCancelledRef.current = false;
      setIsExporting(true);
      setExportProgress(0);

      // Setup Audio Capture (Only once)
      if (!audioCtxRef.current) {
        audioCtxRef.current = new AudioContext();
      }
      const ctx = audioCtxRef.current;
      if (ctx.state === 'suspended') {
        await ctx.resume();
      }
      
      if (!audioSourceRef.current) {
        audioSourceRef.current = ctx.createMediaElementSource(video);
        const dest = ctx.createMediaStreamDestination();
        audioSourceRef.current.connect(dest);
        audioSourceRef.current.connect(ctx.destination);
        
        // Setup MediaRecorder using the combined stream
        const canvasStream = canvas.captureStream(30);
        const combinedStream = new MediaStream([
          ...canvasStream.getVideoTracks(),
          ...dest.stream.getAudioTracks()
        ]);

        const types = [
          'video/mp4;codecs=avc1.42E01E,mp4a.40.2', // H.264 + AAC
          'video/mp4;codecs=avc1,aac',
          'video/mp4',
          'video/webm;codecs=h264,aac',
          'video/webm;codecs=h264',
          'video/webm;codecs=vp9,opus',
          'video/webm'
        ];
        const mimeType = types.find(type => MediaRecorder.isTypeSupported(type)) || 'video/webm';
        
        const recorder = new MediaRecorder(combinedStream, { 
          mimeType,
          videoBitsPerSecond: 10000000, // 10 Mbps for professional quality
          audioBitsPerSecond: 192000   // 192 kbps for clear audio
        });
        recorderRef.current = recorder;
      }

      const recorder = recorderRef.current!;
      const chunks: Blob[] = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      recorder.onstop = () => {
        if (!wasCancelledRef.current && chunks.length > 0) {
          const actualMimeType = recorder.mimeType;
          // Determine extension based on actual used type
          const extension = actualMimeType.includes('mp4') ? 'mp4' : 'webm';
          
          const blob = new Blob(chunks, { type: actualMimeType });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `chromakey-hero-h264-${Date.now()}.${extension}`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
        }
        setIsExporting(false);
        setExportProgress(0);
      };

      // Reset and play
      video.currentTime = 0;
      video.pause();
      
      // Start recording
      recorder.start();
      setIsPlaying(true);
      await video.play();

      // Smoother Progress Monitor using requestAnimationFrame
      const updateProgress = () => {
        if (!isExporting || wasCancelledRef.current) return;
        
        if (video.duration) {
          // Calculate progress based on current time
          const progress = (video.currentTime / video.duration) * 100;
          setExportProgress(prev => {
            const next = Math.max(prev, Math.min(progress, 99.9));
            return next;
          });
        }

        if (!video.ended && isExporting) {
          requestAnimationFrame(updateProgress);
        }
      };
      requestAnimationFrame(updateProgress);

      // Monitor end of video
      const handleEnded = () => {
        setExportProgress(100);
        setTimeout(() => {
          if (recorder.state !== 'inactive') {
            recorder.stop();
          }
        }, 200); // Slight buffer to ensure last frames are captured
        video.removeEventListener('ended', handleEnded);
      };
      video.addEventListener('ended', handleEnded);

    } catch (err) {
      console.error("Export failed:", err);
      setIsExporting(false);
    }
  };

  const cancelExport = () => {
    wasCancelledRef.current = true;
    if (recorderRef.current && recorderRef.current.state !== 'inactive') {
      recorderRef.current.stop();
    }
    if (videoRef.current) {
      videoRef.current.pause();
    }
    setIsPlaying(false);
    setIsExporting(false);
    setExportProgress(0);
  };

  const reset = () => {
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoFile(null);
    setVideoUrl(null);
    setBgImage(null);
    setIsPlaying(false);
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-zinc-100 font-sans selection:bg-emerald-500/30">
      {/* Header */}
      <header className="border-b border-white/10 px-6 py-4 flex items-center justify-between backdrop-blur-md sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-emerald-500 rounded-lg flex items-center justify-center shadow-lg shadow-emerald-500/20">
            <Maximize2 className="text-black size-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">Green screen remover</h1>
            <p className="text-[10px] uppercase tracking-widest text-zinc-500 font-mono">Professional Background Removal</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <button 
            onClick={reset}
            className="p-2 hover:bg-white/5 rounded-full transition-colors text-zinc-400 hover:text-white"
          >
            <RefreshCcw size={20} />
          </button>
        </div>
      </header>

      <main className="max-w-[1400px] mx-auto p-6 lg:p-8">
        {!videoFile || !videoUrl ? (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-2xl mx-auto mt-20"
          >
            <label className="group relative block">
              <input type="file" accept="video/*" onChange={handleFileChange} className="hidden" />
              <div className="border-2 border-dashed border-zinc-800 rounded-3xl p-16 flex flex-col items-center justify-center gap-6 hover:border-emerald-500/50 hover:bg-emerald-500/5 transition-all cursor-pointer overflow-hidden backdrop-blur-sm bg-zinc-900/10">
                <div className="absolute inset-0 bg-gradient-to-b from-emerald-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                <div className="w-20 h-20 bg-zinc-900 rounded-2xl flex items-center justify-center border border-zinc-800 group-hover:scale-110 transition-transform">
                  <Upload className="text-zinc-500 group-hover:text-emerald-500 transition-colors" size={32} />
                </div>
                <div className="text-center">
                  <h3 className="text-2xl font-semibold mb-2">Upload Green Screen Video</h3>
                  <p className="text-zinc-500">MP4, WebM, or OGG supported. No size limit.</p>
                </div>
              </div>
            </label>
          </motion.div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-[1fr_400px] gap-8">
            {/* Preview Section */}
            <div className="space-y-6 lg:sticky lg:top-24 self-start">
              <div className="relative aspect-video bg-zinc-950 rounded-3xl overflow-hidden border border-white/5 shadow-2xl group">
                <video 
                  ref={videoRef}
                  src={videoUrl}
                  className="hidden"
                  loop={!isExporting}
                  playsInline
                  onLoadedMetadata={(e) => {
                    const video = e.currentTarget;
                    setDuration(video.duration);
                    if (canvasRef.current && videoRef.current) {
                      canvasRef.current.width = videoRef.current.videoWidth;
                      canvasRef.current.height = videoRef.current.videoHeight;
                      processFrame(); // Initial draw
                    }
                  }}
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  onTimeUpdate={(e) => {
                    const video = e.currentTarget;
                    setCurrentTime(video.currentTime);
                    if (isExporting && video.duration) {
                      const progress = (video.currentTime / video.duration) * 100;
                      setExportProgress(Math.min(progress, 100));
                    }
                  }}
                />
                <canvas 
                  ref={canvasRef}
                  className="w-full h-full object-contain"
                />

                {/* Export Progress Overlay */}
                <AnimatePresence>
                  {isExporting && (
                    <motion.div 
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="absolute inset-0 z-30 bg-black/80 backdrop-blur-md flex flex-col items-center justify-center p-12 text-center"
                    >
                      <div className="w-full max-w-md space-y-8">
                        <div className="space-y-2">
                          <h3 className="text-2xl font-bold bg-gradient-to-r from-emerald-400 to-emerald-600 bg-clip-text text-transparent">
                            Processing Video
                          </h3>
                          <p className="text-zinc-500 text-sm font-mono uppercase tracking-widest">
                            Rendering pixel data • {Math.round(exportProgress)}%
                          </p>
                        </div>
                        
                        <div className="relative h-2 w-full bg-zinc-900 rounded-full overflow-hidden border border-white/5">
                          <motion.div 
                            className="absolute inset-y-0 left-0 bg-emerald-500 shadow-[0_0_20px_rgba(16,185,129,0.5)]"
                            initial={{ width: 0 }}
                            animate={{ width: `${exportProgress}%` }}
                            transition={{ type: "spring", bounce: 0, duration: 0.2 }}
                          />
                        </div>

                        <button 
                          onClick={cancelExport}
                          className="px-6 py-2 rounded-full border border-white/10 text-[10px] uppercase font-bold tracking-widest text-zinc-500 hover:text-white hover:bg-white/5 transition-all"
                        >
                          Cancel Export
                        </button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Enhanced Playback Controls */}
              <div className="bg-zinc-900/80 backdrop-blur-md rounded-2xl border border-white/5 p-4 space-y-4">
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-[10px] font-mono text-zinc-500 uppercase tracking-widest">
                    <span>{formatTime(currentTime)}</span>
                    <span className="text-emerald-500 font-bold">{Math.round((currentTime / (duration || 1)) * 100)}%</span>
                    <span>{formatTime(duration)}</span>
                  </div>
                  <div className="group/progress relative h-4 flex items-center">
                    <input
                      type="range"
                      min="0"
                      max={duration || 0}
                      step="0.01"
                      value={currentTime}
                      onChange={handleSeek}
                      className="absolute inset-0 w-full h-full opacity-0 z-10 cursor-pointer"
                    />
                    <div className="relative h-1.5 w-full bg-zinc-800 rounded-full overflow-hidden transition-all group-hover/progress:h-2">
                      <div 
                        className="absolute inset-y-0 left-0 bg-emerald-500 shadow-[0_0_12px_rgba(16,185,129,0.4)]"
                        style={{ width: `${(currentTime / (duration || 1)) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-6">
                    <button 
                      onClick={togglePlay}
                      className="w-10 h-10 flex items-center justify-center bg-white text-black rounded-full hover:bg-emerald-500 transition-all shadow-lg hover:scale-105 active:scale-95"
                    >
                      {isPlaying ? <Pause size={20} fill="currentColor" /> : <Play size={20} className="ml-1" fill="currentColor" />}
                    </button>

                    <div className="flex items-center gap-3 group/volume">
                      <button 
                        onClick={() => setIsMuted(!isMuted)}
                        className="text-zinc-400 hover:text-white transition-colors"
                      >
                        {isMuted || volume === 0 ? <VolumeX size={20} /> : <Volume2 size={20} />}
                      </button>
                      <input 
                        type="range"
                        min="0"
                        max="1"
                        step="0.01"
                        value={isMuted ? 0 : volume}
                        onChange={(e) => {
                          setVolume(parseFloat(e.target.value));
                          if (parseFloat(e.target.value) > 0) setIsMuted(false);
                        }}
                        className="w-24 h-1 bg-zinc-800 rounded-full appearance-none accent-emerald-500 cursor-pointer"
                      />
                    </div>
                  </div>

                  <div className="flex items-center gap-4">
                     <button 
                      onClick={toggleFullscreen}
                      className="p-2 text-zinc-400 hover:text-white transition-colors hover:bg-white/5 rounded-lg"
                      title="Toggle Fullscreen"
                    >
                      <Maximize2 size={20} />
                    </button>
                  </div>
                </div>
              </div>

              {/* Status Bar */}
              <div className="flex items-center justify-between p-4 bg-zinc-900/50 rounded-2xl border border-white/5 font-mono text-xs">
                <div className="flex gap-4">
                  <span className="text-zinc-500 uppercase tracking-wider">Resolution: <span className="text-white">{videoRef.current?.videoWidth}x{videoRef.current?.videoHeight}</span></span>
                  <span className="text-zinc-500 uppercase tracking-wider">Status: <span className={isPlaying ? "text-emerald-500" : "text-amber-500"}>{isPlaying ? "Processing" : "Idle"}</span></span>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${isPlaying ? "bg-emerald-500 animate-pulse" : "bg-zinc-700"}`} />
                  <span className="text-zinc-500 opacity-50 uppercase">0.02ms latency</span>
                </div>
              </div>
            </div>

            {/* Sidebar Controls */}
            <aside className="space-y-6 lg:h-[calc(100vh-120px)] lg:overflow-y-auto lg:pr-2 scrollbar-thin scrollbar-track-zinc-900 scrollbar-thumb-zinc-700">
              <section className="bg-zinc-900/50 rounded-3xl border border-white/10 overflow-hidden backdrop-blur-xl">
                <div className="px-6 py-5 border-b border-white/5 flex items-center gap-2">
                  <Settings2 size={18} className="text-emerald-500" />
                  <h2 className="font-semibold tracking-tight">Processing Parameters</h2>
                </div>
                <div className="p-6 space-y-8">
                  {/* Advanced Harmonization Toggles */}
                  {bgMode === 'image' && (
                    <div className="space-y-4 border-b border-white/5 pb-6">
                      <div className="flex items-center justify-between p-3 bg-zinc-950 rounded-2xl border border-white/5">
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-lg ${colorMatch ? 'bg-emerald-500/10 text-emerald-500' : 'bg-zinc-800 text-zinc-400'}`}>
                            <Palette size={16} />
                          </div>
                          <div>
                            <p className="text-xs font-semibold">Color Match</p>
                            <p className="text-[10px] text-zinc-500">Auto environmental color</p>
                          </div>
                        </div>
                        <button
                          onClick={() => setColorMatch(!colorMatch)}
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${
                            colorMatch ? 'bg-emerald-500' : 'bg-zinc-700'
                          }`}
                        >
                          <span
                            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                              colorMatch ? 'translate-x-6' : 'translate-x-1'
                            }`}
                          />
                        </button>
                      </div>

                      <div className="flex items-center justify-between p-3 bg-zinc-950 rounded-2xl border border-white/5">
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-lg ${relighting ? 'bg-emerald-500/10 text-emerald-500' : 'bg-zinc-800 text-zinc-400'}`}>
                            <Sun size={16} />
                          </div>
                          <div>
                            <p className="text-xs font-semibold">Ambient Relighting</p>
                            <p className="text-[10px] text-zinc-500">Simulate background light</p>
                          </div>
                        </div>
                        <button
                          onClick={() => setRelighting(!relighting)}
                          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${
                            relighting ? 'bg-emerald-500' : 'bg-zinc-700'
                          }`}
                        >
                          <span
                            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                              relighting ? 'translate-x-6' : 'translate-x-1'
                            }`}
                          />
                        </button>
                      </div>
                    </div>
                  )}

                  {/* AI Refinement Toggle */}
                  <div className="flex items-center justify-between p-3 bg-zinc-950 rounded-2xl border border-white/5">
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg ${useAISegmentation ? 'bg-emerald-500/10 text-emerald-500' : 'bg-zinc-800 text-zinc-400'} ${isModelLoading ? 'animate-pulse' : ''}`}>
                        <Cpu size={16} />
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <p className="text-xs font-semibold">AI Refinement</p>
                          {isModelLoading && <span className="text-[8px] bg-emerald-500/20 text-emerald-500 px-1 rounded animate-pulse">Loading</span>}
                        </div>
                        <p className="text-[10px] text-zinc-500">Subject recognition mask</p>
                      </div>
                    </div>
                    <button
                      onClick={() => setUseAISegmentation(!useAISegmentation)}
                      disabled={isModelLoading}
                      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${
                        useAISegmentation ? 'bg-emerald-500' : 'bg-zinc-700'
                      } ${isModelLoading ? 'opacity-50 cursor-wait' : ''}`}
                    >
                      <span
                        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                          useAISegmentation ? 'translate-x-6' : 'translate-x-1'
                        }`}
                      />
                    </button>
                  </div>

                  {/* Mask Preview Toggle */}
                  <div className="flex items-center justify-between p-3 bg-zinc-950 rounded-2xl border border-white/5">
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg ${showMask ? 'bg-emerald-500/10 text-emerald-500' : 'bg-zinc-800 text-zinc-400'}`}>
                        {showMask ? <Eye size={16} /> : <EyeOff size={16} />}
                      </div>
                      <div>
                        <p className="text-xs font-semibold">Preview Mask</p>
                        <p className="text-[10px] text-zinc-500">Visualize transparency</p>
                      </div>
                    </div>
                    <button
                      onClick={() => setShowMask(!showMask)}
                      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none ${
                        showMask ? 'bg-emerald-500' : 'bg-zinc-700'
                      }`}
                    >
                      <span
                        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                          showMask ? 'translate-x-6' : 'translate-x-1'
                        }`}
                      />
                    </button>
                  </div>

                  {/* Target Color Selector */}
                  <div className="space-y-3">
                    <div className="flex justify-between items-end">
                      <label className="text-[10px] uppercase font-bold tracking-widest text-zinc-500">Target Hue</label>
                      <div className="text-xs font-mono text-emerald-500">
                        RGB({targetColor.r},{targetColor.g},{targetColor.b})
                      </div>
                    </div>
                    <div className="flex gap-2">
                      {[
                        { r: 0, g: 255, b: 0, name: 'Green' },
                        { r: 0, g: 0, b: 255, name: 'Blue' },
                        { r: 255, g: 0, b: 0, name: 'Red' },
                      ].map((color) => (
                        <button
                          key={color.name}
                          onClick={() => setTargetColor({ r: color.r, g: color.g, b: color.b })}
                          className={`flex-1 h-10 rounded-xl transition-all border-2 ${
                            targetColor.r === color.r && targetColor.g === color.g && targetColor.b === color.b
                              ? 'border-white scale-105'
                              : 'border-transparent opacity-50 hover:opacity-100'
                          }`}
                          style={{ backgroundColor: `rgb(${color.r}, ${color.g}, ${color.b})` }}
                          title={color.name}
                        />
                      ))}
                    </div>
                  </div>

                  {/* Range Sliders */}
                  <div className="space-y-6">
                    <div className="space-y-4">
                      <div className="flex justify-between text-xs">
                        <label className="text-[10px] uppercase font-bold tracking-widest text-zinc-500">Similarity Threshold</label>
                        <span className="font-mono">{tolerance}</span>
                      </div>
                      <input 
                        type="range" 
                        min="0" 
                        max="200" 
                        value={tolerance}
                        onChange={(e) => setTolerance(Number(e.target.value))}
                        className="w-full accent-emerald-500"
                      />
                    </div>

                    <div className="space-y-4">
                      <div className="flex justify-between text-xs">
                        <label className="text-[10px] uppercase font-bold tracking-widest text-zinc-500">Edge Smoothing</label>
                        <span className="font-mono">{smoothing}</span>
                      </div>
                      <input 
                        type="range" 
                        min="0" 
                        max="100" 
                        value={smoothing}
                        onChange={(e) => setSmoothing(Number(e.target.value))}
                        className="w-full accent-emerald-500"
                      />
                    </div>

                    <div className="space-y-4">
                      <div className="flex justify-between text-xs">
                        <div className="flex items-center gap-2">
                           <label className="text-[10px] uppercase font-bold tracking-widest text-zinc-500">Light Wrapping</label>
                           <Sparkles size={10} className="text-emerald-500" />
                        </div>
                        <span className="font-mono">{lightWrap}%</span>
                      </div>
                      <input 
                        type="range" 
                        min="0" 
                        max="100" 
                        value={lightWrap}
                        onChange={(e) => setLightWrap(Number(e.target.value))}
                        className="w-full accent-emerald-500"
                      />
                      <p className="text-[9px] text-zinc-600 leading-tight">Blends background color into subject edges for realistic integration.</p>
                    </div>

                    <div className="space-y-4 border-t border-white/5 pt-4">
                      <div className="flex justify-between items-center text-xs">
                        <label className="text-[10px] uppercase font-bold tracking-widest text-zinc-500">Processing Resolution</label>
                        <span className="font-mono text-emerald-500">{Math.round(processingScale * 100)}%</span>
                      </div>
                      <div className="flex gap-2">
                        {[0.5, 0.75, 1.0].map((scale) => (
                          <button
                            key={scale}
                            onClick={() => setProcessingScale(scale)}
                            className={`flex-1 py-1 px-2 rounded-lg text-[10px] font-bold border transition-all ${
                              processingScale === scale 
                                ? 'bg-emerald-500 border-emerald-500 text-black' 
                                : 'bg-zinc-950 border-white/5 text-zinc-500 hover:text-white'
                            }`}
                          >
                            {scale === 0.5 ? 'Low' : scale === 0.75 ? 'Mid' : 'Native'}
                          </button>
                        ))}
                      </div>
                      <p className="text-[9px] text-zinc-600">Lower resolution significantly improves processing speed.</p>
                    </div>

                    {useAISegmentation && (
                      <div className="space-y-4 pt-4 border-t border-white/5">
                        <div className="flex justify-between items-center text-xs">
                          <label className="text-[10px] uppercase font-bold tracking-widest text-zinc-500">AI Refresh Rate</label>
                          <span className="font-mono text-emerald-500">Every {aiFrequency}f</span>
                        </div>
                        <div className="flex gap-2">
                          {[1, 3, 5, 10].map((freq) => (
                            <button
                              key={freq}
                              onClick={() => setAiFrequency(freq)}
                              className={`flex-1 py-1 px-2 rounded-lg text-[10px] font-bold border transition-all ${
                                aiFrequency === freq 
                                  ? 'bg-emerald-500 border-emerald-500 text-black' 
                                  : 'bg-zinc-950 border-white/5 text-zinc-500 hover:text-white'
                              }`}
                            >
                              {freq === 1 ? 'High' : freq === 3 ? 'Mid' : freq === 5 ? 'Low' : 'Min'}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="space-y-4">
                      <div className="flex justify-between text-xs">
                        <label className="text-[10px] uppercase font-bold tracking-widest text-zinc-500">Despill Strength</label>
                        <span className="font-mono">{despillStrength}%</span>
                      </div>
                      <input 
                        type="range" 
                        min="0" 
                        max="100" 
                        value={despillStrength}
                        onChange={(e) => setDespillStrength(Number(e.target.value))}
                        className="w-full accent-emerald-500"
                      />
                    </div>

                    <div className="space-y-4 border-t border-white/5 pt-4">
                      <div className="flex justify-between text-xs">
                        <label className="text-[10px] uppercase font-bold tracking-widest text-zinc-500">Matte Shrink</label>
                        <span className="font-mono">{matteShrink}px</span>
                      </div>
                      <input 
                        type="range" 
                        min="0" 
                        max="10" 
                        step="0.5"
                        value={matteShrink}
                        onChange={(e) => setMatteShrink(Number(e.target.value))}
                        className="w-full accent-emerald-500"
                      />
                    </div>

                    <div className="space-y-4">
                      <div className="flex justify-between text-xs">
                        <label className="text-[10px] uppercase font-bold tracking-widest text-zinc-500">Edge Softness</label>
                        <span className="font-mono">{matteFeather}</span>
                      </div>
                      <input 
                        type="range" 
                        min="0" 
                        max="50" 
                        value={matteFeather}
                        onChange={(e) => setMatteFeather(Number(e.target.value))}
                        className="w-full accent-emerald-500"
                      />
                    </div>
                  </div>
                </div>
              </section>

              <section className="bg-zinc-900/50 rounded-3xl border border-white/10 overflow-hidden backdrop-blur-xl">
                <div className="px-6 py-5 border-b border-white/5 flex items-center gap-2">
                  <ImageIcon size={18} className="text-emerald-500" />
                  <h2 className="font-semibold tracking-tight">Background Context</h2>
                </div>
                <div className="p-6 space-y-6">
                  {/* BG Mode Selector */}
                  <div className="flex bg-zinc-950 p-1 rounded-2xl border border-white/5">
                    {(['transparent', 'color', 'image'] as const).map((mode) => (
                      <button
                        key={mode}
                        onClick={() => setBgMode(mode)}
                        className={`flex-1 py-2 text-[10px] uppercase font-bold tracking-tighter rounded-xl transition-all ${
                          bgMode === mode ? 'bg-zinc-800 text-white' : 'text-zinc-500 hover:text-white'
                        }`}
                      >
                        {mode}
                      </button>
                    ))}
                  </div>

                  {/* Mode Specific Controls */}
                  <AnimatePresence mode="wait">
                    {bgMode === 'color' && (
                      <motion.div 
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="space-y-3"
                      >
                         <label className="text-[10px] uppercase font-bold tracking-widest text-zinc-500">Matte Color</label>
                         <div className="flex gap-3">
                           <input 
                            type="color" 
                            value={bgColor} 
                            onChange={(e) => setBgColor(e.target.value)}
                            className="w-12 h-12 rounded-xl border-none outline-none cursor-pointer bg-transparent"
                           />
                           <input 
                            type="text" 
                            value={bgColor} 
                            onChange={(e) => setBgColor(e.target.value)}
                            className="flex-1 bg-zinc-950 border border-white/5 rounded-xl px-4 text-sm font-mono focus:outline-none focus:border-emerald-500/50 transition-colors uppercase"
                           />
                         </div>
                      </motion.div>
                    )}

                    {bgMode === 'image' && (
                      <motion.div 
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="space-y-3"
                      >
                        <label className="text-[10px] uppercase font-bold tracking-widest text-zinc-500">Background Image</label>
                        <label className="block">
                          <input type="file" accept="image/*" onChange={handleBgImageChange} className="hidden" />
                          <div className="border border-dashed border-white/10 rounded-2xl p-6 text-center hover:bg-white/5 transition-all cursor-pointer">
                            {bgImage ? (
                              <div className="relative group rounded-xl overflow-hidden aspect-video">
                                <img src={bgImage} className="w-full h-full object-cover" />
                                <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity gap-2">
                                  <span className="text-[10px] uppercase font-bold tracking-widest text-white px-4 py-1.5 border border-white/20 rounded-full bg-white/10 backdrop-blur-sm group-hover:border-white/50 transition-all">Replace</span>
                                  <button 
                                    onClick={(e) => {
                                      e.preventDefault();
                                      e.stopPropagation();
                                      setBgImage(null);
                                    }}
                                    className="text-[9px] uppercase font-bold tracking-widest text-red-400 hover:text-red-300 transition-colors"
                                  >
                                    Remove Image
                                  </button>
                                </div>
                              </div>
                            ) : (
                              <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">Select Image</span>
                            )}
                          </div>
                        </label>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </section>

              {/* Action Buttons */}
              <div className="flex gap-3">
                <button 
                  onClick={startExport}
                  disabled={!videoUrl || isExporting}
                  className={`flex-1 py-4 rounded-3xl font-bold uppercase text-[11px] tracking-[0.2em] transition-all hover:scale-[1.02] flex items-center justify-center gap-2 ${
                    isExporting 
                    ? "bg-zinc-800 text-zinc-500 cursor-not-allowed" 
                    : "bg-white text-black hover:bg-emerald-500 shadow-xl shadow-white/5"
                  }`}
                >
                  {isExporting ? (
                    <>
                      <RefreshCcw size={16} className="animate-spin" />
                      Exporting...
                    </>
                  ) : (
                    <>
                      <Download size={16} />
                      Download Video
                    </>
                  )}
                </button>
                <button 
                  onClick={reset}
                  className="p-4 bg-zinc-900 border border-white/5 text-red-500 rounded-3xl hover:bg-red-500/10 transition-colors"
                >
                  <Trash2 size={18} />
                </button>
              </div>
            </aside>
          </div>
        )}
      </main>

      {/* Decorative Elements */}
      <div className="fixed bottom-0 left-0 w-full h-[30vh] bg-gradient-to-t from-emerald-500/5 to-transparent pointer-events-none -z-10" />
      <div className="fixed top-0 right-0 w-[50vw] h-screen bg-gradient-to-l from-emerald-500/5 to-transparent pointer-events-none -z-10" />
    </div>
  );
}
