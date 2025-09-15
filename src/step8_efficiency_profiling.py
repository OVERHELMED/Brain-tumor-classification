"""
Step 8: Efficiency Profiling for Brain MRI Classification Pipeline

This script profiles the computational efficiency of the MobileNetV2-based
brain MRI classification pipeline, measuring parameters, FLOPs, latency,
memory usage, and input size trade-offs.
"""

import os
import json
import time
import psutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try to import FLOP counting tools
try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False
    print("ptflops not available, will use alternative FLOP counting")

try:
    from fvcore.nn import flop_count, parameter_count
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("fvcore not available, will use alternative FLOP counting")


class EfficiencyProfiler:
    """
    Comprehensive efficiency profiler for the brain MRI classification pipeline.
    """
    
    def __init__(self, features_dir, labels_path):
        """Initialize the efficiency profiler."""
        self.features_dir = features_dir
        self.labels_path = labels_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu_device = torch.device("cpu")
        
        # Load labels info
        with open(labels_path, 'r') as f:
            self.labels_info = json.load(f)
        
        self.class_names = self.labels_info['class_names']
        self.num_classes = len(self.class_names)
        
        print(f"Device: {self.device}")
        print(f"Classes: {self.class_names}")
        
        # Setup models and preprocessing
        self._setup_models()
        
    def _setup_models(self):
        """Setup the MobileNetV2 model and preprocessing transforms."""
        # Create MobileNetV2 model (feature extractor only)
        self.mobilenetv2 = timm.create_model(
            'mobilenetv2_100',
            pretrained=True,
            num_classes=0,  # No classifier
            global_pool='avg'
        )
        self.mobilenetv2.eval()
        
        # Setup preprocessing transforms for different input sizes
        self.transforms = {
            224: transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ]),
            192: transforms.Compose([
                transforms.Resize((192, 192)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ]),
            160: transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        }
        
        # Load trained classifier
        self._load_classifier()
        
    def _load_classifier(self):
        """Load the trained classifier and scaler."""
        # Load features and labels for scaler fitting
        X_train = np.load(os.path.join(self.features_dir, "train_X.npy"))
        y_train = np.load(os.path.join(self.features_dir, "train_y.npy"))
        X_val = np.load(os.path.join(self.features_dir, "val_X.npy"))
        y_val = np.load(os.path.join(self.features_dir, "val_y.npy"))
        
        # Standardize features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train classifier
        self.classifier = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            class_weight='balanced',
            random_state=42,
            C=10,
            max_iter=2000
        )
        
        # Fit on combined train+val
        X_combined = np.vstack([X_train_scaled, X_val_scaled])
        y_combined = np.concatenate([y_train, y_val])
        self.classifier.fit(X_combined, y_combined)
        
        # Load calibrated classifier
        self.calibrated_classifier = CalibratedClassifierCV(
            self.classifier, method='sigmoid', cv='prefit'
        )
        self.calibrated_classifier.fit(X_val_scaled, y_val)
        
    def count_parameters_and_flops(self, input_size=224):
        """
        Count model parameters and FLOPs.
        
        Args:
            input_size: Input image size
            
        Returns:
            Dictionary with parameter and FLOP counts
        """
        print(f"Counting parameters and FLOPs for input size {input_size}x{input_size}...")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.mobilenetv2.parameters())
        trainable_params = sum(p.numel() for p in self.mobilenetv2.parameters() if p.requires_grad)
        
        # Count FLOPs using available tools
        flops_count = None
        
        if PTFLOPS_AVAILABLE:
            try:
                # Use ptflops
                flops, params = get_model_complexity_info(
                    self.mobilenetv2, 
                    (3, input_size, input_size),
                    print_per_layer_stat=False,
                    verbose=False
                )
                flops_count = flops
                print(f"ptflops result: {flops}")
            except Exception as e:
                print(f"ptflops failed: {e}")
                
        if FVCORE_AVAILABLE and flops_count is None:
            try:
                # Use fvcore
                dummy_input = torch.randn(1, 3, input_size, input_size)
                flops_dict, _ = flop_count(self.mobilenetv2, (dummy_input,))
                flops_count = flops_dict
                print(f"fvcore result: {flops_count}")
            except Exception as e:
                print(f"fvcore failed: {e}")
        
        if flops_count is None:
            # Manual estimation based on MobileNetV2 architecture
            # MobileNetV2-1.0 at 224x224 is approximately 300 MFLOPs
            scale_factor = (input_size / 224) ** 2
            estimated_flops = 300 * scale_factor
            flops_count = f"{estimated_flops:.1f}M"
            print(f"Using manual estimation: {flops_count} MFLOPs")
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'flops': flops_count,
            'input_size': input_size
        }
        
    def benchmark_latency(self, num_runs=500, warmup_runs=50):
        """
        Benchmark inference latency on CPU and GPU.
        
        Args:
            num_runs: Number of inference runs for timing
            warmup_runs: Number of warmup runs
            
        Returns:
            Dictionary with latency measurements
        """
        print(f"Benchmarking latency with {num_runs} runs (warmup: {warmup_runs})...")
        
        # Create dummy input
        dummy_image = torch.randn(1, 3, 224, 224)
        
        results = {}
        
        # Test on CPU
        print("Testing CPU latency...")
        cpu_model = self.mobilenetv2.to(self.cpu_device)
        cpu_latencies = []
        
        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = cpu_model(dummy_image)
        
        # Benchmark
        for _ in tqdm(range(num_runs), desc="CPU benchmarking"):
            start_time = time.time()
            with torch.no_grad():
                features = cpu_model(dummy_image)
                # Apply classifier
                features_np = features.numpy()
                features_scaled = self.scaler.transform(features_np)
                _ = self.classifier.predict_proba(features_scaled)
            end_time = time.time()
            cpu_latencies.append(end_time - start_time)
        
        # Test on GPU if available
        gpu_latencies = []
        if self.device.type == 'cuda':
            print("Testing GPU latency...")
            gpu_model = self.mobilenetv2.to(self.device)
            gpu_dummy = dummy_image.to(self.device)
            
            # Warmup
            for _ in range(warmup_runs):
                with torch.no_grad():
                    _ = gpu_model(gpu_dummy)
            
            # Benchmark
            for _ in tqdm(range(num_runs), desc="GPU benchmarking"):
                start_time = time.time()
                with torch.no_grad():
                    features = gpu_model(gpu_dummy)
                    # Apply classifier (move back to CPU for sklearn)
                    features_np = features.cpu().numpy()
                    features_scaled = self.scaler.transform(features_np)
                    _ = self.classifier.predict_proba(features_scaled)
                end_time = time.time()
                gpu_latencies.append(end_time - start_time)
        
        # Calculate statistics
        results['cpu'] = {
            'mean_latency_ms': np.mean(cpu_latencies) * 1000,
            'std_latency_ms': np.std(cpu_latencies) * 1000,
            'throughput_ips': 1.0 / np.mean(cpu_latencies),
            'min_latency_ms': np.min(cpu_latencies) * 1000,
            'max_latency_ms': np.max(cpu_latencies) * 1000
        }
        
        if gpu_latencies:
            results['gpu'] = {
                'mean_latency_ms': np.mean(gpu_latencies) * 1000,
                'std_latency_ms': np.std(gpu_latencies) * 1000,
                'throughput_ips': 1.0 / np.mean(gpu_latencies),
                'min_latency_ms': np.min(gpu_latencies) * 1000,
                'max_latency_ms': np.max(gpu_latencies) * 1000
            }
        
        return results
        
    def measure_memory_footprint(self):
        """Measure memory usage and model sizes."""
        print("Measuring memory footprint...")
        
        # Model sizes
        model_size_mb = 0
        for param in self.mobilenetv2.parameters():
            model_size_mb += param.numel() * param.element_size() / (1024 * 1024)
        
        # Classifier size (approximate)
        classifier_size_mb = (
            self.classifier.coef_.nbytes + 
            self.classifier.intercept_.nbytes
        ) / (1024 * 1024)
        
        # Feature cache sizes
        feature_sizes = {}
        for split in ['train', 'val', 'test']:
            feature_file = os.path.join(self.features_dir, f"{split}_X.npy")
            if os.path.exists(feature_file):
                feature_array = np.load(feature_file)
                feature_sizes[f'{split}_features_mb'] = feature_array.nbytes / (1024 * 1024)
        
        # Memory usage during inference
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Run inference to measure peak memory
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            _ = self.mobilenetv2(dummy_input)
        
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        inference_memory_mb = memory_after - memory_before
        
        return {
            'mobilenetv2_size_mb': model_size_mb,
            'classifier_size_mb': classifier_size_mb,
            'total_model_size_mb': model_size_mb + classifier_size_mb,
            'inference_memory_mb': inference_memory_mb,
            'feature_cache_sizes': feature_sizes
        }
        
    def input_size_ablation(self, sizes=[224, 192, 160]):
        """
        Compare performance at different input sizes.
        
        Args:
            sizes: List of input sizes to test
            
        Returns:
            Dictionary with performance and efficiency metrics for each size
        """
        print(f"Running input size ablation for sizes: {sizes}")
        
        results = {}
        
        for size in sizes:
            print(f"\nTesting input size {size}x{size}...")
            
            # Create models for different input sizes
            model = timm.create_model(
                'mobilenetv2_100',
                pretrained=True,
                num_classes=0,
                global_pool='avg'
            )
            model.eval()
            
            # Count parameters and FLOPs
            param_flop_info = self.count_parameters_and_flops(size)
            
            # Benchmark latency
            dummy_input = torch.randn(1, 3, size, size)
            
            # CPU latency
            cpu_times = []
            model_cpu = model.to(self.cpu_device)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model_cpu(dummy_input)
            
            # Benchmark
            for _ in range(100):
                start_time = time.time()
                with torch.no_grad():
                    features = model_cpu(dummy_input)
                    features_np = features.numpy()
                    features_scaled = self.scaler.transform(features_np)
                    _ = self.classifier.predict_proba(features_scaled)
                end_time = time.time()
                cpu_times.append(end_time - start_time)
            
            # GPU latency if available
            gpu_times = []
            if self.device.type == 'cuda':
                model_gpu = model.to(self.device)
                dummy_gpu = dummy_input.to(self.device)
                
                # Warmup
                for _ in range(10):
                    with torch.no_grad():
                        _ = model_gpu(dummy_gpu)
                
                # Benchmark
                for _ in range(100):
                    start_time = time.time()
                    with torch.no_grad():
                        features = model_gpu(dummy_gpu)
                        features_np = features.cpu().numpy()
                        features_scaled = self.scaler.transform(features_np)
                        _ = self.classifier.predict_proba(features_scaled)
                    end_time = time.time()
                    gpu_times.append(end_time - start_time)
            
            results[size] = {
                'parameters': param_flop_info['total_parameters'],
                'flops': param_flop_info['flops'],
                'cpu_latency_ms': np.mean(cpu_times) * 1000,
                'cpu_throughput_ips': 1.0 / np.mean(cpu_times),
                'gpu_latency_ms': np.mean(gpu_times) * 1000 if gpu_times else None,
                'gpu_throughput_ips': 1.0 / np.mean(gpu_times) if gpu_times else None,
                'speedup_vs_224_cpu': results.get(224, {}).get('cpu_latency_ms', 1) / (np.mean(cpu_times) * 1000) if size != 224 else 1.0,
                'speedup_vs_224_gpu': results.get(224, {}).get('gpu_latency_ms', 1) / (np.mean(gpu_times) * 1000) if size != 224 and gpu_times else 1.0
            }
        
        return results
        
    def run_complete_profiling(self):
        """Run complete efficiency profiling pipeline."""
        print("=== Starting Complete Efficiency Profiling ===\n")
        
        results = {}
        
        # 1. Parameters and FLOPs
        print("1. Measuring parameters and FLOPs...")
        results['model_complexity'] = self.count_parameters_and_flops(224)
        
        # 2. Latency benchmarking
        print("\n2. Benchmarking inference latency...")
        results['latency_benchmark'] = self.benchmark_latency(num_runs=500, warmup_runs=50)
        
        # 3. Memory footprint
        print("\n3. Measuring memory footprint...")
        results['memory_footprint'] = self.measure_memory_footprint()
        
        # 4. Input size ablation
        print("\n4. Running input size ablation...")
        results['input_size_ablation'] = self.input_size_ablation([224, 192, 160])
        
        # 5. Training efficiency
        print("\n5. Measuring training efficiency...")
        results['training_efficiency'] = self._measure_training_efficiency()
        
        return results
        
    def _measure_training_efficiency(self):
        """Measure training time and efficiency."""
        # Load a subset of data for timing
        X_train = np.load(os.path.join(self.features_dir, "train_X.npy"))[:1000]
        y_train = np.load(os.path.join(self.features_dir, "train_y.npy"))[:1000]
        
        # Time classifier training
        start_time = time.time()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        classifier = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            class_weight='balanced',
            random_state=42,
            C=10,
            max_iter=2000
        )
        classifier.fit(X_scaled, y_train)
        training_time = time.time() - start_time
        
        return {
            'classifier_training_time_seconds': training_time,
            'training_samples': len(X_train),
            'training_samples_per_second': len(X_train) / training_time
        }


def main():
    """Main function for Step 8: Efficiency Profiling."""
    print("=== Step 8: Efficiency Profiling ===\n")
    
    # Setup paths
    features_dir = "data/brainmri_4c/features/mobilenetv2"
    labels_path = "data/brainmri_4c/labels.json"
    
    # Initialize profiler
    profiler = EfficiencyProfiler(features_dir, labels_path)
    
    # Run complete profiling
    results = profiler.run_complete_profiling()
    
    # Save results
    print("\n--- Saving Efficiency Profiling Results ---")
    os.makedirs('experiments/results', exist_ok=True)
    
    with open('experiments/results/efficiency_profiling_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print(f"\n=== Efficiency Profiling Summary ===")
    
    # Model complexity
    complexity = results['model_complexity']
    print(f"📊 Model Complexity:")
    print(f"  Parameters: {complexity['total_parameters']:,}")
    print(f"  FLOPs: {complexity['flops']}")
    
    # Latency
    latency = results['latency_benchmark']
    print(f"\n⏱️  Inference Latency:")
    print(f"  CPU: {latency['cpu']['mean_latency_ms']:.2f} ± {latency['cpu']['std_latency_ms']:.2f} ms")
    print(f"  CPU Throughput: {latency['cpu']['throughput_ips']:.2f} images/sec")
    if 'gpu' in latency:
        print(f"  GPU: {latency['gpu']['mean_latency_ms']:.2f} ± {latency['gpu']['std_latency_ms']:.2f} ms")
        print(f"  GPU Throughput: {latency['gpu']['throughput_ips']:.2f} images/sec")
    
    # Memory
    memory = results['memory_footprint']
    print(f"\n💾 Memory Usage:")
    print(f"  Model Size: {memory['total_model_size_mb']:.2f} MB")
    print(f"  Inference Memory: {memory['inference_memory_mb']:.2f} MB")
    
    # Input size comparison
    ablation = results['input_size_ablation']
    print(f"\n📏 Input Size Comparison:")
    for size, metrics in ablation.items():
        print(f"  {size}x{size}: {metrics['cpu_latency_ms']:.2f}ms CPU, {metrics['cpu_throughput_ips']:.2f} IPS")
        if metrics['gpu_latency_ms']:
            print(f"    GPU: {metrics['gpu_latency_ms']:.2f}ms, {metrics['gpu_throughput_ips']:.2f} IPS")
    
    # Training efficiency
    training = results['training_efficiency']
    print(f"\n🏋️  Training Efficiency:")
    print(f"  Classifier Training: {training['classifier_training_time_seconds']:.2f}s")
    print(f"  Training Speed: {training['training_samples_per_second']:.0f} samples/sec")
    
    print(f"\n📁 Results saved to: experiments/results/efficiency_profiling_results.json")
    print(f"🎉 Efficiency profiling complete! Ready for paper integration.")


if __name__ == "__main__":
    main()
