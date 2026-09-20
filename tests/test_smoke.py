"""确保 backend 包可被导入（脚手架自检）。"""


def test_app_package_importable():
    import app  # noqa: F401


def test_tools_subpackages_importable():
    import app.api  # noqa: F401
    import app.agent  # noqa: F401
    import app.tools.ocp  # noqa: F401
    import app.tools.sql  # noqa: F401
