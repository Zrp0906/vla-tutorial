r"""初始化采集阶段的 Python 包命名空间。

输入：
    - Python 导入 smol_workflow.p03_data_collection 或其子模块。

输出：
    - 提供包命名空间和模块文档；本文件不导出业务函数，不创建数据、模型、回放或报告。

独立运行：
    - 不需要独立运行；__init__.py 是 Python 包初始化文件，不是执行顺序中的步骤。
    - 仅导入此文件不会执行训练；具体操作由同阶段编号入口调用。

运行参数：
    - 无命令行参数；包名用于 import，不是要执行的脚本。

用法：
    ```bash
    cd /Users/zrp/Desktop/lora/smol
    conda run -n embodiedai python -c "import smol_workflow.p03_data_collection"
    ```

被谁依赖：
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p00_pipeline/_impl/p00_cli.py`（导入子模块时初始化此包）。
    - `/Users/zrp/Desktop/lora/smol/smol_workflow/p04_data_processing/_impl/p04_data.py`（导入子模块时初始化此包）。"""
