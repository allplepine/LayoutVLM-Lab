"""Main pipeline for Layout+VLM document processing."""

import os
import logging
import traceback
from typing import List, Dict, Any
from multiprocessing import Process, Queue, Lock, Manager, set_start_method
from threading import Thread
from tqdm import tqdm
import copy
from .config import Config, load_config
from .core.registry import LAYOUT_REGISTRY, VLM_REGISTRY
from .layout.base import BaseLayoutDetector
from .vlm.base import BaseVLM
from .utils import crop_by_boxes


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def layout_worker_process(
    image_queue: Queue,
    vlm_queue: Queue,
    config: Config,
    progress: Dict,
    lock: Lock,
    gpu_id: int
):
    """Layout worker process - PRODUCER.
    
    Runs Layout detection and puts results in vlm_queue.
    """
    import queue
    
    # Override device config for this specific process
    # We copy the layout config to avoid modifying the global shared config object
    # (though multiprocessing shares read-only usually, it's safer)
    layout_cfg = config.layout.config.copy()
    layout_cfg["device"] = f"gpu:{gpu_id}"
    # Build Layout instance from Registry
    layout = LAYOUT_REGISTRY.build({
        "type": config.layout.type,
        "config": layout_cfg
    })
    
    while True:
        try:
            task = image_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        
        if task is None:
            break
        
        image_file = task
        img_path = os.path.join(config.input.image_root, image_file)
        
        if not os.path.isfile(img_path):
            logger.error("Image not found: %s", img_path)
            with lock:
                progress['failed'] += 1
            continue
        
        try:
            data = layout.detect(img_path)
            vlm_queue.put((image_file, data, img_path))
            with lock:
                progress['layout_done'] += 1
        except Exception:
            logger.error("Layout failed for %s:\n%s", image_file, traceback.format_exc())
            with lock:
                progress['failed'] += 1


def vlm_worker_thread(
    vlm_queue: Queue,
    result_queue: Queue,
    config: Config,
    vlm: BaseVLM,
    layout_cls: BaseLayoutDetector,
    progress: Dict,
    lock: Lock
):
    """VLM worker thread - CONSUMER.
    
    Consumes Layout results and processes with VLM.
    """
    import queue
    
    # Labels to ignore (from config)
    ignore_labels = set(config.layout.label_mapping.get("ignore", []))
    
    # Reverse mapping: generic metric -> [model-specific labels]
    # We need to map model specific label to metric label
    label_to_metric = {}
    for metric, labels in config.layout.label_mapping.items():
        if metric == "ignore": continue
        for lb in labels:
            label_to_metric[lb] = metric
            
    # Metrics enabled for VLM
    enabled_metrics = set(config.vlm.metrics)
    
    # Image labels (for saving)
    image_labels = set(config.layout.label_mapping.get("image", []))
    
    # Prepare layout config for static methods
    # json2md needs label_mapping which is separate in config.layout
    layout_static_config = config.layout.config.copy()
    layout_static_config["label_mapping"] = config.layout.label_mapping
    
    while True:
        try:
            task = vlm_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        
        if task is None:
            break
        
        image_file, data, img_path = task
        base_name = os.path.splitext(image_file)[0]
        blocks = data.get("parsing_res_list", [])
        counts = {}
        
        # Process blocks with VLM
        for idx, block in enumerate(blocks):
            label = block.get("block_label", "")
            
            if label in ignore_labels:
                continue
                
            # Get metric type
            # Default to "ocr" if not implicitly ignored or mapped
            metric_type = label_to_metric.get(label, "ocr")
            # Skip if this metric is not enabled in VLM config
            if metric_type not in enabled_metrics:
                continue
            
            bbox = block.get("block_bbox")
            if not bbox:
                continue
            
            try:
                cropped = crop_by_boxes(img_path, [bbox])
                if not cropped:
                    continue
                # Recognize using metric_type which maps to prompt
                content = vlm.recognize(cropped[0], metric_type)
                
                # Post-process content using Layout-specific logic
                # We need the original label here
                content = layout_cls.post_process(content, label, layout_static_config)
                
                blocks[idx]["block_content"] = content
                
                # Count by metric type
                counts[metric_type] = counts.get(metric_type, 0) + 1
            except Exception:
                logger.error("VLM error for %s block %d:\n%s", 
                            image_file, idx, traceback.format_exc())
        
        # Save images for image-type blocks
        if config.output.save_images:
            for block in blocks:
                label = block.get("block_label")
                if label in image_labels and block.get("block_bbox"):
                    try:
                        bbox = block["block_bbox"]
                        x1, y1, x2, y2 = map(int, bbox)
                        cropped = crop_by_boxes(img_path, [[x1, y1, x2, y2]])
                        if cropped:
                            crop_name = f"{base_name}_{block['block_id']}.jpg"
                            crop_save_path = os.path.join(config.imgs_dir, crop_name)
                            cropped[0].save(crop_save_path, "JPEG", quality=95)
                    except Exception:
                        pass
        
        # Save outputs
        try:
            # Save Markdown (Delegated to Layout handler)
            data_copy = copy.deepcopy(data)
            md_content = layout_cls.json2md(data_copy, layout_static_config)
            md_path = os.path.join(config.results_dir, f"{base_name}.md")
            layout_cls.save_md(md_path, md_content)
            
            # Save JSON
            json_path = os.path.join(config.json_dir, f"{base_name}_res.json")
            layout_cls.save_json(json_path, data)

            result_queue.put(counts)
        except Exception:
            logger.error("Save failed for %s:\n%s", image_file, traceback.format_exc())
            result_queue.put({})
        
        with lock:
            progress['vlm_done'] += 1


def run_pipeline(config: Config, image_files: List[str] = None) -> Dict[str, int]:
    """Run the Layout+VLM pipeline."""
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Get image files
    if image_files is None:
        extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
        all_files = [
            f for f in os.listdir(config.input.image_root)
            if os.path.splitext(f)[1].lower() in extensions
        ]
        
        # Filter processed
        if config.input.filter_processed:
            processed = set(os.listdir(config.json_dir))
            image_files = [
                f for f in all_files
                if f"{os.path.splitext(f)[0]}_res.json" not in processed
            ]
        else:
            image_files = all_files
    
    if not image_files:
        logger.warning("No images to process")
        return {}
    
    total = len(image_files)
    logger.info("Processing %d images", total)
    
    # Determing GPU deployment strategy
    # "0,1,2" -> [0, 1, 2]
    # nums_per_gpu -> workers per gpu
    
    device_str = config.layout.config.get("device_list", "0")
    device_list = [int(x.strip()) for x in device_str.split(",") if x.strip()]
    nums_per_gpu = int(config.layout.config.get("nums_per_gpu", 1))
    
    total_workers = len(device_list) * nums_per_gpu
    vlm_workers = config.pipeline.vlm_workers
    
    logger.info("Starting %d Layout processes (%d GPUs * %d), %d VLM threads", 
                total_workers, len(device_list), nums_per_gpu, vlm_workers)
    
    manager = Manager()
    progress = manager.dict({'layout_done': 0, 'vlm_done': 0, 'failed': 0})
    
    image_queue = Queue()
    vlm_queue = Queue(maxsize=config.pipeline.vlm_queue_max_size)
    result_queue = Queue()
    lock = Lock()
    
    # Fill image queue
    for f in image_files:
        image_queue.put(f)
    for _ in range(total_workers):
        image_queue.put(None)
    
    # Start Layout processes
    layout_procs = []
    
    for gpu_idx in device_list:
        for _ in range(nums_per_gpu):
            p = Process(
                target=layout_worker_process,
                args=(image_queue, vlm_queue, config, progress, lock, gpu_idx)
            )
            p.daemon = True
            p.start()
            layout_procs.append(p)
    
    # Build VLM instance from Registry
    vlm = VLM_REGISTRY.build({
        "type": config.vlm.type,
        "config": config.vlm.config
    })
    
    # Get Layout CLASS from Registry (no instantiation needed for static methods)
    layout_cls = LAYOUT_REGISTRY.get(config.layout.type)
    
    # Start VLM threads
    vlm_threads = []
    for _ in range(vlm_workers):
        t = Thread(
            target=vlm_worker_thread,
            args=(vlm_queue, result_queue, config, vlm, layout_cls, progress, lock)
        )
        t.daemon = True
        t.start()
        vlm_threads.append(t)
    
    # Progress monitoring
    import queue as queue_module
    counts = {}
    
    with tqdm(total=total, desc="Processing", unit="file", dynamic_ncols=True) as pbar:
        last_done = 0
        while progress['vlm_done'] < total - progress['failed']:
            current = progress['vlm_done']
            if current > last_done:
                pbar.update(current - last_done)
                last_done = current
            
            pbar.set_description(
                f"Layout:{progress['layout_done']}/{total} VLM:{current}"
            )
            
            # Collect results
            while True:
                try:
                    result = result_queue.get_nowait()
                    for k, v in result.items():
                        counts[k] = counts.get(k, 0) + v
                except queue_module.Empty:
                    break
            
            import time
            time.sleep(0.1)
            
            if progress['layout_done'] + progress['failed'] >= total:
                if progress['vlm_done'] >= total - progress['failed']:
                    break
        
        pbar.update(total - pbar.n)
    
    # Final drain of result_queue to catch any stragglers
    while True:
        try:
            result = result_queue.get_nowait()
            for k, v in result.items():
                counts[k] = counts.get(k, 0) + v
        except queue_module.Empty:
            break
    
    # Cleanup
    for p in layout_procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    
    for _ in range(vlm_workers):
        vlm_queue.put(None)
    
    for t in vlm_threads:
        t.join(timeout=5)
    
    logger.info("Done. Results saved to: %s", config.run_dir)
    logger.info("Counts: %s", counts)
    
    return counts
