# main.py

import argparse
import logging
import pandas as pd
import torch

from defaults import get_cfg_defaults
from registry import PREPROCESSING_STEPS, FEATURE_EXTRACTORS, MODELS

# Side-effect imports (register components)
import preprocessing.setup  # Registers preprocessing steps
import models.models         # Registers models
# import core.features       # If/when needed

import ipdb

import torch.optim as optim

from train import train_sklearn


def parse_args():
    parser = argparse.ArgumentParser(description="Main training and evaluation script.")
    parser.add_argument("--cfg", dest="config_file", required=True, help="Path to config file")
    return parser.parse_args()


def load_config(path):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(path)
    cfg.freeze()
    return cfg


def configure_pipeline(cfg_section, registry):
    logging.info(f"Configuring pipeline for '{registry._name}'...")
    configured_steps = []
    for step_config in cfg_section:
        step_name = list(step_config.keys())[0]
        params = step_config.get(step_name) or {}
        step_builder = registry.get(step_name)
        step_function = step_builder(params)
        configured_steps.append((step_name, step_function))
    return configured_steps


def apply_pipeline(data, steps):
    logging.info("Applying pipeline...")
    for name, func in steps:
        logging.info(f"  -> Applying step: {name}")
        data = func(data)
    return data


def build_model1(cfg):
    ipdb.set_trace()
    logging.info("Building model...")
    model_builder = MODELS.get(cfg.model.type)
    if cfg.model.type == "HuggingFace":
        return model_builder(cfg.model)
    else:
        return model_builder(cfg.model.params)
    
def build_model(cfg):
    logging.info("Building Model...")
    model_builder = MODELS.get(cfg.model.type)
    
    if cfg.model.type == 'custom':
        model = model_builder(cfg.model)
    else:
        model = model_builder(cfg.model.get('params', {}))
        
    logging.info(f"Model '{cfg.model.type}' built.")
    return model


def load_data(path):
    logging.info(f"Loading data from {path}...")
    # Replace with actual loading logic
    return pd.Series([
        "I LOVE this new #product!! It's amazing :) #excited",
        "I HATE waiting in line... so boring :( #fail"
    ])

def create_optimizer(cfg, model):
    ipdb.set_trace()
    opt_type = cfg['optimizer']['type']
    params = cfg['optimizer']['params']
    
    validate_class_name(torch.optim, opt_type, "optimizer")
    opt_class = getattr(torch.optim, opt_type)
    return opt_class(model.parameters(), **params)

def create_scheduler(cfg, optimizer):
    sched_type = cfg['scheduler']['type']
    params = cfg['scheduler']['params']
    
    validate_class_name(torch.optim.lr_scheduler, sched_type, "scheduler")
    sched_class = getattr(torch.optim.lr_scheduler, sched_type)
    return sched_class(optimizer, **params)

def validate_class_name(module, class_name, module_name):
    """Check if class_name exists in module. Raise error with friendly message if not."""
    if not hasattr(module, class_name):
        available = [cls for cls in dir(module) if not cls.startswith('_')]
        raise ValueError(
            f"Invalid {module_name} type '{class_name}'. "
            f"Please use the exact class name as in PyTorch, e.g. one of: {available}"
        )


def main():
    args = parse_args()
    cfg = load_config(args.config_file)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info(f"Loaded configuration from {args.config_file}")

    # Pipeline
    preprocessing_steps = configure_pipeline(cfg.preprocessing, PREPROCESSING_STEPS)

    # Data
    raw_data = load_data(cfg.dataset)
    processed_data = apply_pipeline(raw_data, preprocessing_steps)

    logging.info("\n--- Original Data ---")
    print(raw_data)

    logging.info("\n--- Processed Data ---")
    print(processed_data[0])
    print(processed_data[1])

    # Model
    model = build_model(cfg)
    logging.info("\n✅ Model build finished successfully!")
    logging.info(model)

    train_sklearn(raw_data, model, cfg)

    #optimizer = create_optimizer(cfg, model)
    #scheduler = create_scheduler(cfg, optimizer)

    #logging.info(f"\n✅ Optimizer: {optimizer}")
    #logging.info(f"✅ Scheduler: {scheduler}")


if __name__ == "__main__":
    main()
