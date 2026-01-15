# -*- coding: utf-8 -*-
"""
@File    :   run_tests.py
@Time    :   2026/01/15 18:00:00
@Author  :   Xiaopeng Gu
@Version :   1.0
@Desc    :   运行所有测试用例并生成HTML测试报告
"""

import os
import unittest
import sys
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # 尝试导入HTMLTestRunner
    import HTMLTestRunner
    use_html_runner = True
except ImportError:
    # 如果没有安装HTMLTestRunner，使用默认的TextTestRunner
    use_html_runner = False
    print("HTMLTestRunner未安装，将使用默认的TextTestRunner")


def run_all_tests():
    """运行所有测试用例"""
    # 获取当前目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建测试套件
    suite = unittest.TestSuite()
    
    # 发现并添加所有测试用例
    loader = unittest.TestLoader()
    test_dir = os.path.join(current_dir, "utils")
    
    # 发现所有以test_开头的测试文件
    test_files = loader.discover(test_dir, pattern="test_*.py")
    
    # 将所有测试用例添加到套件中
    for test_file in test_files:
        suite.addTest(test_file)
    
    # 创建报告目录
    report_dir = os.path.join(current_dir, "report")
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    # 生成报告文件名
    now = datetime.now()
    report_file_name = f"test_report_{now.strftime('%Y%m%d_%H%M%S')}.html"
    report_file_path = os.path.join(report_dir, report_file_name)
    
    # 运行测试并生成报告
    if use_html_runner:
        # 使用HTMLTestRunner生成HTML报告
        with open(report_file_path, "wb") as f:
            runner = HTMLTestRunner.HTMLTestRunner(
                stream=f,
                title="Utils模块测试报告",
                description="测试utils文件夹下所有工具类的功能",
                verbosity=2
            )
            result = runner.run(suite)
    else:
        # 使用默认的TextTestRunner
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # 将测试结果写入文本文件
        text_report_path = os.path.join(report_dir, f"test_report_{now.strftime('%Y%m%d_%H%M%S')}.txt")
        with open(text_report_path, "w") as f:
            f.write(f"测试运行时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试用例总数: {result.testsRun}\n")
            f.write(f"成功测试数: {result.testsRun - len(result.failures) - len(result.errors)}\n")
            f.write(f"失败测试数: {len(result.failures)}\n")
            f.write(f"错误测试数: {len(result.errors)}\n\n")
            
            if result.failures:
                f.write("\n失败的测试用例:\n")
                for failure in result.failures:
                    f.write(f"{failure[0]}: {failure[1]}\n\n")
            
            if result.errors:
                f.write("\n发生错误的测试用例:\n")
                for error in result.errors:
                    f.write(f"{error[0]}: {error[1]}\n\n")
    
    # 输出测试结果摘要
    print("\n" + "="*50)
    print("测试结果摘要")
    print("="*50)
    print(f"测试用例总数: {result.testsRun}")
    print(f"成功测试数: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败测试数: {len(result.failures)}")
    print(f"错误测试数: {len(result.errors)}")
    
    if use_html_runner:
        print(f"HTML测试报告已生成: {report_file_path}")
    else:
        print(f"文本测试报告已生成: {text_report_path}")
    
    return result


if __name__ == "__main__":
    # 运行所有测试用例
    run_all_tests()
