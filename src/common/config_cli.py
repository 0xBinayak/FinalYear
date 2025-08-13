#!/usr/bin/env python3
"""
Configuration Management CLI Tool
Provides command-line interface for managing configurations, A/B tests, and validation
"""

import argparse
import json
import yaml
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from config import ConfigManager, ABTestManager


class ConfigCLI:
    """Command-line interface for configuration management"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.ab_test_manager = ABTestManager()
    
    def validate_config(self, environment: str = None) -> bool:
        """Validate configuration for specified environment"""
        try:
            config = self.config_manager.load_config(environment)
            is_valid = self.config_manager.validate_config(config)
            
            if is_valid:
                print(f"✅ Configuration for '{config.environment}' is valid")
                return True
            else:
                print(f"❌ Configuration for '{config.environment}' is invalid")
                return False
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False
    
    def show_config(self, environment: str = None, format_type: str = "yaml"):
        """Display current configuration"""
        try:
            config = self.config_manager.load_config(environment)
            config_dict = config.__dict__
            
            # Convert nested dataclasses to dicts
            for key, value in config_dict.items():
                if hasattr(value, '__dict__'):
                    config_dict[key] = value.__dict__
            
            if format_type.lower() == "json":
                print(json.dumps(config_dict, indent=2, default=str))
            else:
                print(yaml.dump(config_dict, default_flow_style=False))
                
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
    
    def create_ab_test(self, test_name: str, variants_file: str, 
                      traffic_split: float = 0.5, duration_hours: int = 24):
        """Create a new A/B test"""
        try:
            with open(variants_file, 'r') as f:
                if variants_file.endswith('.json'):
                    variants = json.load(f)
                else:
                    variants = yaml.safe_load(f)
            
            test_id = self.ab_test_manager.create_test(
                test_name, variants, traffic_split, duration_hours
            )
            
            print(f"✅ Created A/B test '{test_name}' with ID: {test_id}")
            print(f"   Traffic split: {traffic_split * 100}% A / {(1-traffic_split) * 100}% B")
            print(f"   Duration: {duration_hours} hours")
            
        except Exception as e:
            print(f"❌ Failed to create A/B test: {e}")
    
    def show_ab_test_results(self, test_name: str):
        """Show A/B test results"""
        try:
            results = self.ab_test_manager.get_test_results(test_name)
            
            if "error" in results:
                print(f"❌ {results['error']}")
                return
            
            print(f"📊 A/B Test Results: {test_name}")
            print(f"Total samples: {results['total_samples']}")
            print()
            
            for variant, stats in results['variant_stats'].items():
                print(f"Variant {variant}:")
                print(f"  Count: {stats['count']}")
                print(f"  Mean: {stats['mean']:.4f}")
                print(f"  Min: {stats['min']:.4f}")
                print(f"  Max: {stats['max']:.4f}")
                print()
                
        except Exception as e:
            print(f"❌ Failed to get A/B test results: {e}")
    
    def stop_ab_test(self, test_name: str):
        """Stop an active A/B test"""
        try:
            results = self.ab_test_manager.stop_test(test_name)
            print(f"🛑 Stopped A/B test: {test_name}")
            self.show_ab_test_results(test_name)
            
        except Exception as e:
            print(f"❌ Failed to stop A/B test: {e}")
    
    def generate_config_template(self, environment: str, output_file: str = None):
        """Generate configuration template for specified environment"""
        template = {
            "environment": environment,
            "debug": environment == "development",
            "config_version": "1.0.0",
            "last_updated": datetime.now().isoformat(),
            "database": {
                "host": "localhost" if environment == "development" else f"postgres-{environment}",
                "port": 5432,
                "database": f"federated_pipeline_{environment}",
                "username": "postgres",
                "password": ""
            },
            "network": {
                "host": "0.0.0.0",
                "port": 8000,
                "max_connections": 100 if environment == "development" else 1000,
                "timeout": 30
            },
            "federated_learning": {
                "aggregation_strategy": "fedavg",
                "min_clients": 1 if environment == "development" else 10,
                "max_clients": 10 if environment == "development" else 1000,
                "rounds": 5 if environment == "development" else 50,
                "local_epochs": 5,
                "learning_rate": 0.01,
                "batch_size": 32
            },
            "privacy": {
                "enable_differential_privacy": environment == "production",
                "epsilon": 1.0 if environment == "development" else 0.5,
                "delta": 1e-5,
                "noise_multiplier": 1.0
            },
            "monitoring": {
                "enable_metrics": True,
                "metrics_port": 9090,
                "log_level": "DEBUG" if environment == "development" else "INFO",
                "enable_tracing": environment != "production"
            },
            "ab_test": {
                "enabled": False,
                "test_name": "",
                "variant": "A",
                "traffic_split": 0.5,
                "parameters": {}
            }
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                yaml.dump(template, f, default_flow_style=False)
            print(f"✅ Generated configuration template: {output_file}")
        else:
            print(yaml.dump(template, default_flow_style=False))
    
    def check_config_drift(self):
        """Check for configuration drift between environments"""
        environments = ["development", "staging", "production"]
        configs = {}
        
        print("🔍 Checking configuration drift between environments...")
        print()
        
        # Load all environment configs
        for env in environments:
            try:
                config = self.config_manager.load_config(env)
                configs[env] = config.__dict__
            except Exception as e:
                print(f"⚠️  Could not load {env} config: {e}")
                continue
        
        # Compare configurations
        if len(configs) < 2:
            print("❌ Need at least 2 environment configs to compare")
            return
        
        base_env = list(configs.keys())[0]
        base_config = configs[base_env]
        
        for env, config in configs.items():
            if env == base_env:
                continue
            
            print(f"📋 Comparing {base_env} vs {env}:")
            self._compare_configs(base_config, config, "")
            print()
    
    def _compare_configs(self, config1: Dict[str, Any], config2: Dict[str, Any], prefix: str):
        """Recursively compare two configuration dictionaries"""
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in sorted(all_keys):
            full_key = f"{prefix}.{key}" if prefix else key
            
            if key not in config1:
                print(f"  + {full_key}: missing in first config")
            elif key not in config2:
                print(f"  - {full_key}: missing in second config")
            elif isinstance(config1[key], dict) and isinstance(config2[key], dict):
                self._compare_configs(config1[key], config2[key], full_key)
            elif config1[key] != config2[key]:
                print(f"  ≠ {full_key}: {config1[key]} → {config2[key]}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Configuration Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument("--env", help="Environment to validate")
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show configuration")
    show_parser.add_argument("--env", help="Environment to show")
    show_parser.add_argument("--format", choices=["yaml", "json"], default="yaml")
    
    # A/B test commands
    ab_create_parser = subparsers.add_parser("ab-create", help="Create A/B test")
    ab_create_parser.add_argument("test_name", help="Name of the A/B test")
    ab_create_parser.add_argument("variants_file", help="File containing variant configurations")
    ab_create_parser.add_argument("--split", type=float, default=0.5, help="Traffic split (0.0-1.0)")
    ab_create_parser.add_argument("--duration", type=int, default=24, help="Duration in hours")
    
    ab_results_parser = subparsers.add_parser("ab-results", help="Show A/B test results")
    ab_results_parser.add_argument("test_name", help="Name of the A/B test")
    
    ab_stop_parser = subparsers.add_parser("ab-stop", help="Stop A/B test")
    ab_stop_parser.add_argument("test_name", help="Name of the A/B test")
    
    # Template command
    template_parser = subparsers.add_parser("template", help="Generate configuration template")
    template_parser.add_argument("environment", help="Environment name")
    template_parser.add_argument("--output", help="Output file path")
    
    # Drift check command
    subparsers.add_parser("drift", help="Check configuration drift between environments")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = ConfigCLI()
    
    try:
        if args.command == "validate":
            success = cli.validate_config(args.env)
            sys.exit(0 if success else 1)
        elif args.command == "show":
            cli.show_config(args.env, args.format)
        elif args.command == "ab-create":
            cli.create_ab_test(args.test_name, args.variants_file, args.split, args.duration)
        elif args.command == "ab-results":
            cli.show_ab_test_results(args.test_name)
        elif args.command == "ab-stop":
            cli.stop_ab_test(args.test_name)
        elif args.command == "template":
            cli.generate_config_template(args.environment, args.output)
        elif args.command == "drift":
            cli.check_config_drift()
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()