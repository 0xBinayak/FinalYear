#!/usr/bin/env python3
"""
Script to run all demonstrations in the Advanced Federated Pipeline system.
"""

import asyncio
import argparse
import sys
import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class DemoRunner:
    """Manages execution of multiple demonstrations."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        # Define available demonstrations
        self.demonstrations = {
            "field_testing": [
                "demos/field_testing/basic_field_test.py"
            ],
            "real_world": [
                "demos/real_world/federated_learning_demo.py"
            ],
            "signal_processing": [
                "demos/signal_processing/signal_processing_demo.py"
            ],
            "integration": [
                "demos/integration/full_system_demo.py"
            ],
            "security": [
                "demos/security/privacy_security_demo.py"
            ]
        }
    
    def print_banner(self):
        """Print runner banner."""
        print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║              DEMONSTRATION SUITE RUNNER                                      ║
    ║                                                                              ║
    ║  Automated execution of all system demonstrations                            ║
    ║  • Field Testing                                                             ║
    ║  • Real-World Scenarios                                                      ║
    ║  • Signal Processing                                                         ║
    ║  • System Integration                                                        ║
    ║  • Security & Privacy                                                        ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
        """)
    
    async def run_demonstration(self, demo_path: str, timeout: int = 300) -> Dict:
        """Run a single demonstration."""
        demo_name = Path(demo_path).stem
        print(f"\n🚀 Running {demo_name}...")
        print(f"   Path: {demo_path}")
        
        start_time = time.time()
        
        try:
            # Prepare command
            cmd = [sys.executable, demo_path]
            
            # Add common arguments for automated execution
            if "field_testing" in demo_path:
                cmd.extend(["--duration", "5", "--no-viz"])
            elif "real_world" in demo_path:
                cmd.extend(["--quick"])
            elif "integration" in demo_path:
                cmd.extend(["--timeout", "60"])
            
            # Run the demonstration
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path(__file__).parent.parent
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Demonstration timed out after {timeout} seconds")
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Process results
            success = process.returncode == 0
            
            result = {
                "demo_name": demo_name,
                "demo_path": demo_path,
                "success": success,
                "return_code": process.returncode,
                "duration": duration,
                "stdout": stdout.decode('utf-8', errors='ignore'),
                "stderr": stderr.decode('utf-8', errors='ignore'),
                "timestamp": datetime.now().isoformat()
            }
            
            if success:
                print(f"   ✅ {demo_name} completed successfully ({duration:.1f}s)")
            else:
                print(f"   ❌ {demo_name} failed (return code: {process.returncode})")
                if stderr:
                    print(f"   Error: {stderr.decode('utf-8', errors='ignore')[:200]}...")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"   💥 {demo_name} crashed: {e}")
            
            return {
                "demo_name": demo_name,
                "demo_path": demo_path,
                "success": False,
                "return_code": -1,
                "duration": duration,
                "stdout": "",
                "stderr": str(e),
                "timestamp": datetime.now().isoformat(),
                "exception": str(e)
            }
    
    async def run_category(self, category: str, timeout: int = 300) -> List[Dict]:
        """Run all demonstrations in a category."""
        if category not in self.demonstrations:
            raise ValueError(f"Unknown category: {category}")
        
        print(f"\n📂 Running {category.replace('_', ' ').title()} Demonstrations")
        print("="*60)
        
        results = []
        demos = self.demonstrations[category]
        
        for demo_path in demos:
            if not Path(demo_path).exists():
                print(f"   ⚠️  Demo not found: {demo_path}")
                continue
            
            result = await self.run_demonstration(demo_path, timeout)
            results.append(result)
        
        return results
    
    async def run_all_categories(self, categories: List[str], timeout: int = 300) -> Dict:
        """Run demonstrations from multiple categories."""
        self.start_time = time.time()
        all_results = {}
        
        for category in categories:
            try:
                category_results = await self.run_category(category, timeout)
                all_results[category] = category_results
            except Exception as e:
                print(f"❌ Failed to run category {category}: {e}")
                all_results[category] = []
        
        self.end_time = time.time()
        return all_results
    
    def generate_summary_report(self, results: Dict) -> Dict:
        """Generate a summary report of all demonstrations."""
        total_demos = 0
        successful_demos = 0
        total_duration = 0
        
        category_summaries = {}
        
        for category, category_results in results.items():
            category_total = len(category_results)
            category_successful = sum(1 for r in category_results if r["success"])
            category_duration = sum(r["duration"] for r in category_results)
            
            category_summaries[category] = {
                "total": category_total,
                "successful": category_successful,
                "failed": category_total - category_successful,
                "success_rate": category_successful / category_total if category_total > 0 else 0,
                "total_duration": category_duration
            }
            
            total_demos += category_total
            successful_demos += category_successful
            total_duration += category_duration
        
        overall_duration = self.end_time - self.start_time if self.start_time and self.end_time else total_duration
        
        summary = {
            "execution_info": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
                "total_duration": overall_duration,
                "execution_date": datetime.now().isoformat()
            },
            "overall_summary": {
                "total_demonstrations": total_demos,
                "successful_demonstrations": successful_demos,
                "failed_demonstrations": total_demos - successful_demos,
                "overall_success_rate": successful_demos / total_demos if total_demos > 0 else 0,
                "total_demo_duration": total_duration
            },
            "category_summaries": category_summaries,
            "detailed_results": results
        }
        
        return summary
    
    def print_summary(self, summary: Dict):
        """Print a formatted summary of results."""
        print(f"\n{'='*80}")
        print(f"DEMONSTRATION SUITE RESULTS")
        print(f"{'='*80}")
        
        overall = summary["overall_summary"]
        print(f"Total Demonstrations: {overall['total_demonstrations']}")
        print(f"Successful: {overall['successful_demonstrations']}")
        print(f"Failed: {overall['failed_demonstrations']}")
        print(f"Success Rate: {overall['overall_success_rate']:.1%}")
        print(f"Total Duration: {overall['total_demo_duration']:.1f}s")
        
        if summary["execution_info"]["total_duration"]:
            print(f"Wall Clock Time: {summary['execution_info']['total_duration']:.1f}s")
        
        print(f"\nCategory Breakdown:")
        for category, cat_summary in summary["category_summaries"].items():
            success_rate = cat_summary["success_rate"]
            status_icon = "✅" if success_rate == 1.0 else "⚠️" if success_rate > 0.5 else "❌"
            
            print(f"  {status_icon} {category.replace('_', ' ').title()}:")
            print(f"     Success: {cat_summary['successful']}/{cat_summary['total']} ({success_rate:.1%})")
            print(f"     Duration: {cat_summary['total_duration']:.1f}s")
        
        # Show failed demonstrations
        failed_demos = []
        for category, category_results in summary["detailed_results"].items():
            for result in category_results:
                if not result["success"]:
                    failed_demos.append(f"{category}/{result['demo_name']}")
        
        if failed_demos:
            print(f"\nFailed Demonstrations:")
            for demo in failed_demos:
                print(f"  ❌ {demo}")
        
        print(f"\n{'='*80}")
    
    def save_results(self, summary: Dict, output_file: str):
        """Save results to a JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"📁 Results saved to: {output_path}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Advanced Federated Pipeline demonstrations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run all demonstrations
  %(prog)s --category security                # Run only security demos
  %(prog)s --category field_testing real_world # Run specific categories
  %(prog)s --timeout 600 --output results.json # Custom timeout and output
        """
    )
    
    parser.add_argument(
        "--category", 
        nargs="+",
        choices=["field_testing", "real_world", "signal_processing", "integration", "security"],
        help="Specific categories to run (default: all)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per demonstration in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="demo_output/demo_suite_results.json",
        help="Output file for results (default: demo_output/demo_suite_results.json)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing summary"
    )
    
    args = parser.parse_args()
    
    # Determine categories to run
    if args.category:
        categories = args.category
    else:
        categories = ["field_testing", "real_world", "signal_processing", "integration", "security"]
    
    # Create runner
    runner = DemoRunner(args.config)
    runner.print_banner()
    
    print(f"🎯 Running demonstrations for categories: {', '.join(categories)}")
    print(f"⏱️  Timeout per demonstration: {args.timeout}s")
    
    try:
        # Run demonstrations
        results = await runner.run_all_categories(categories, args.timeout)
        
        # Generate summary
        summary = runner.generate_summary_report(results)
        
        # Print summary
        if not args.no_summary:
            runner.print_summary(summary)
        
        # Save results
        runner.save_results(summary, args.output)
        
        # Determine exit code
        overall_success_rate = summary["overall_summary"]["overall_success_rate"]
        
        if overall_success_rate == 1.0:
            print("🎉 All demonstrations completed successfully!")
            return 0
        elif overall_success_rate >= 0.8:
            print("⚠️  Most demonstrations completed successfully")
            return 0
        else:
            print("❌ Many demonstrations failed")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration suite interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)