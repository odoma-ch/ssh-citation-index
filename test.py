#!/usr/bin/env python3
"""
LLM Structured Output Test Script
Tests how max_output parameter affects structured output quality
"""

import json
import time
import re
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd

# Import the project's LLM client
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from citation_index.llm.client import LLMClient, DeepSeekClient

class LLMStructuredOutputTester:
    def __init__(self, endpoint: str, model: str, api_key: str = None):
        """
        Initialize the tester with LLM client configuration
        
        Args:
            endpoint (str): LLM API endpoint
            model (str): Model to test
            api_key (str): API key for authentication
        """
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        
        # Initialize the LLM client
        self.llm_client = LLMClient(endpoint=endpoint, model=model, api_key=api_key)
        
        self.test_prompt = self._get_test_prompt()
        self.results = []
        
    def _get_test_prompt(self) -> str:
        """Generate the test prompt for structured output"""
        return """You are a data analyst tasked with creating a comprehensive company analysis report. Please generate a detailed structured output in JSON format for the following company analysis:

**Company**: TechCorp Solutions
**Industry**: Software Development
**Founded**: 2019
**Employees**: 250

Your response must include ALL of the following sections in valid JSON format:

1. **company_overview**: Basic company information
2. **financial_metrics**: Revenue, profit margins, growth rates
3. **market_analysis**: Market size, competition, positioning
4. **strengths**: List of 8-10 key strengths
5. **weaknesses**: List of 6-8 key weaknesses  
6. **opportunities**: List of 8-10 market opportunities
7. **threats**: List of 6-8 potential threats
8. **strategic_recommendations**: 12-15 detailed actionable recommendations
9. **risk_assessment**: Detailed risk analysis with mitigation strategies
10. **growth_projections**: 5-year growth forecast with quarterly breakdowns
11. **competitive_analysis**: Analysis of top 5 competitors
12. **technology_stack**: Current and recommended technologies
13. **hr_analysis**: Staffing needs, culture assessment, retention strategies
14. **marketing_strategy**: Go-to-market plans, customer acquisition
15. **operational_efficiency**: Process improvements and cost optimization

Each section should be detailed with specific data points, explanations, and actionable insights. Use realistic but fictional data throughout.

Format your entire response as a single, valid JSON object."""

    def is_valid_json(self, text: str) -> bool:
        """Check if the response is valid JSON"""
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False
    
    def count_json_sections(self, text: str) -> int:
        """Count how many of the 15 expected sections are present"""
        expected_sections = [
            "company_overview", "financial_metrics", "market_analysis",
            "strengths", "weaknesses", "opportunities", "threats",
            "strategic_recommendations", "risk_assessment", "growth_projections",
            "competitive_analysis", "technology_stack", "hr_analysis",
            "marketing_strategy", "operational_efficiency"
        ]
        
        count = 0
        for section in expected_sections:
            if f'"{section}"' in text or f"'{section}'" in text:
                count += 1
        return count
    
    def analyze_content_quality(self, text: str) -> Dict[str, Any]:
        """Analyze the quality and completeness of the response"""
        analysis = {
            "char_count": len(text),
            "word_count": len(text.split()),
            "is_valid_json": self.is_valid_json(text),
            "sections_found": self.count_json_sections(text),
            "sections_percentage": (self.count_json_sections(text) / 15) * 100,
            "ends_abruptly": not text.strip().endswith('}'),
            "has_truncation_indicators": any(indicator in text.lower() for indicator in ['...', 'truncated', 'cut off', 'incomplete'])
        }
        
        # Try to parse JSON and get more detailed analysis
        if analysis["is_valid_json"]:
            try:
                parsed = json.loads(text)
                analysis["json_keys"] = list(parsed.keys()) if isinstance(parsed, dict) else []
                analysis["json_structure_depth"] = self._get_json_depth(parsed)
            except:
                analysis["json_keys"] = []
                analysis["json_structure_depth"] = 0
        else:
            analysis["json_keys"] = []
            analysis["json_structure_depth"] = 0
            
        return analysis
    
    def _get_json_depth(self, obj: Any, depth: int = 0) -> int:
        """Calculate the maximum depth of JSON structure"""
        if isinstance(obj, dict):
            return max([self._get_json_depth(v, depth + 1) for v in obj.values()] + [depth])
        elif isinstance(obj, list):
            return max([self._get_json_depth(item, depth + 1) for item in obj] + [depth])
        else:
            return depth
    
    def run_single_test(self, max_tokens: int, temperature: float = 0.7) -> Dict[str, Any]:
        """Run a single test with specified max_tokens"""
        print(f"Running test with max_tokens={max_tokens}...")
        
        try:
            # Use the project's LLM client
            response_text = self.llm_client.call(
                prompt=self.test_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Since we don't have finish_reason from our client, we'll infer it
            finish_reason = "stop"  # Default assumption
            
            # Analyze the response
            analysis = self.analyze_content_quality(response_text)
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "endpoint": self.endpoint,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "finish_reason": finish_reason,
                "response_text": response_text,
                **analysis
            }
            
            self.results.append(result)
            return result
            
        except Exception as e:
            print(f"Error in test with max_tokens={max_tokens}: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "endpoint": self.endpoint,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "error": str(e),
                "response_text": "",
                "char_count": 0,
                "word_count": 0,
                "is_valid_json": False,
                "sections_found": 0,
                "sections_percentage": 0.0
            }
    
    def run_full_test_suite(self, token_limits: List[int] = None) -> List[Dict[str, Any]]:
        """Run the complete test suite with different token limits"""
        if token_limits is None:
            token_limits = [8000, 4000, 2000, 1000, 500]
        
        print(f"Starting test suite with model: {self.model}")
        print(f"Endpoint: {self.endpoint}")
        print(f"Token limits to test: {token_limits}")
        print("-" * 60)
        
        for max_tokens in token_limits:
            self.run_single_test(max_tokens)
            time.sleep(1)  # Rate limiting
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report"""
        if not self.results:
            return "No test results available. Run tests first."
        
        report = []
        report.append("LLM STRUCTURED OUTPUT TEST RESULTS")
        report.append("=" * 50)
        report.append(f"Model: {self.model}")
        report.append(f"Endpoint: {self.endpoint}")
        report.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Tests: {len(self.results)}")
        report.append("")
        
        # Summary table
        report.append("SUMMARY TABLE")
        report.append("-" * 30)
        report.append(f"{'Max Tokens':<12} {'Valid JSON':<12} {'Sections':<10} {'Quality':<10} {'Finish Reason':<15}")
        report.append("-" * 70)
        
        for result in self.results:
            if 'error' not in result:
                quality_score = "High" if result['sections_percentage'] > 80 else \
                               "Medium" if result['sections_percentage'] > 50 else \
                               "Low" if result['sections_percentage'] > 20 else "Very Low"
                
                report.append(f"{result['max_tokens']:<12} {str(result['is_valid_json']):<12} "
                             f"{result['sections_found']}/15{'':<4} {quality_score:<10} "
                             f"{result['finish_reason']:<15}")
        
        report.append("")
        
        # Detailed analysis
        report.append("DETAILED ANALYSIS")
        report.append("-" * 30)
        
        for result in self.results:
            if 'error' not in result:
                report.append(f"\nTest: {result['max_tokens']} tokens")
                report.append(f"  Valid JSON: {result['is_valid_json']}")
                report.append(f"  Sections Found: {result['sections_found']}/15 ({result['sections_percentage']:.1f}%)")
                report.append(f"  Response Length: {result['char_count']} chars, {result['word_count']} words")
                report.append(f"  Finish Reason: {result['finish_reason']}")
                report.append(f"  Ends Abruptly: {result['ends_abruptly']}")
                report.append(f"  Has Truncation Indicators: {result['has_truncation_indicators']}")
                if result['is_valid_json']:
                    report.append(f"  JSON Keys: {len(result['json_keys'])}")
                    report.append(f"  JSON Depth: {result['json_structure_depth']}")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = None):
        """Save results to JSON and CSV files"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"llm_test_results_{timestamp}"
        
        # Save detailed results as JSON
        with open(f"{filename}.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary as CSV
        summary_data = []
        for result in self.results:
            if 'error' not in result:
                summary_data.append({
                    'max_tokens': result['max_tokens'],
                    'is_valid_json': result['is_valid_json'],
                    'sections_found': result['sections_found'],
                    'sections_percentage': result['sections_percentage'],
                    'char_count': result['char_count'],
                    'word_count': result['word_count'],
                    'finish_reason': result['finish_reason'],
                    'ends_abruptly': result['ends_abruptly']
                })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(f"{filename}_summary.csv", index=False)
        
        print(f"Results saved to {filename}.json and {filename}_summary.csv")


def main():
    """Main function to run the test"""
    # Configuration - Update these with your actual values
    ENDPOINT = "https://api.deepseek.com/v1"  # or your preferred endpoint
    MODEL = "deepseek-chat"  # or your preferred model
    API_KEY = "your-api-key-here"  # Replace with your actual API key
    
    # Test parameters
    TOKEN_LIMITS = [8000, 4000, 2000, 1000, 500, 200]  # Different limits to test
    
    # Initialize tester with your project's LLM client
    tester = LLMStructuredOutputTester(ENDPOINT, MODEL, API_KEY)
    
    # Run tests
    print("Starting LLM Structured Output Tests...")
    results = tester.run_full_test_suite(TOKEN_LIMITS)
    
    # Generate and print report
    print("\n" + "="*60)
    print(tester.generate_report())
    
    # Save results
    tester.save_results()
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()