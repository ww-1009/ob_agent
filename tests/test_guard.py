import pytest

from app.tools.sql.guard import ReadOnlyViolation, assert_read_only, assert_safe_identifier

ALLOWED = [
    "select * from orders",
    "SELECT id FROM orders WHERE status='PAID'",
    "  -- comment\nselect 1",
    "show tables",
    "SHOW CREATE TABLE orders",
    "explain select * from orders",
    "explain format=tree select * from orders",
    "explain analyze select * from orders",  # analyze 只是 EXPLAIN 的修饰词
    "describe orders",
    "desc orders",
    "with x as (select 1) select * from x",
    "select * from orders;",  # 尾部单分号允许
    "select * from orders;;",  # 尾部空语句允许
    "/* multi\n line */ select 1",
    "# hi\nselect 1",
    "select * from orders -- trailing comment",
    # 字面量与引号标识符里出现写关键字，属于正常查询（旧实现会误拒）
    "select 'for update'",
    "select * from orders where cond='x;y'",
    "select * from orders where note='into outfile /tmp/x'",
    "select `update` from orders",
    'select "delete" from orders',
]

DENIED = [
    "update orders set status='x'",
    "DELETE FROM orders",
    "insert into orders values (1)",
    "drop table orders",
    "alter table orders add column c int",
    "truncate table orders",
    "select * from orders for update",
    "select * from orders for share",
    "select * from orders; drop table orders",  # 多语句
    "call some_proc()",
    "create index idx on orders(id)",
    "select * from orders for /* x */ update",
    "select * from orders lock /* x */ in share mode",
    "select * from orders into outfile '/tmp/x'",
    "select 'a' into dumpfile '/tmp/y'",
    "SELECT 1 INTO OUTFILE '/var/lib/mysql/x'",  # 大小写变体
    "SELECT 1 INTO /* c */ OUTFILE '/tmp/f'",  # 块注释变体
]

# 实跑确认过的旧实现绕过路径：这些曾经全部放行，逐条固化防回归
BYPASSES_NOW_DENIED = [
    "select * from orders for --c\n update",  # `--c` 在 MySQL 里不是注释，Oracle 里是
    "select * from orders for -- c\n update",  # 行注释把关键字切开
    "select * from orders for #c\n update",  # `#` 行注释（MySQL）
    "select * from orders into --c\n outfile '/tmp/x'",
    "with x as (select 1) delete from orders",  # CTE 之后的 DML
    "with x as (select 1) insert into orders select * from x",
    "with x as (select 1) update orders set status='x'",
    "explain analyze delete from orders",
    "explain delete from orders",
    "explain plan for update orders set status='x'",
]

# fail closed：识别不出内层语句时一律拒绝
DENIED_FAIL_CLOSED = [
    "explain",
    "explain analyze",
    "; drop table orders",
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allows_readonly(sql):
    assert_read_only(sql)  # 不应抛异常


@pytest.mark.parametrize("sql", DENIED + BYPASSES_NOW_DENIED + DENIED_FAIL_CLOSED)
def test_denies_writes(sql):
    with pytest.raises(ReadOnlyViolation):
        assert_read_only(sql)


def test_rejects_empty_sql():
    for sql in ("", "   ", "-- only comment", "/* only comment */", ";"):
        with pytest.raises(ReadOnlyViolation):
            assert_read_only(sql)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("orders", "orders"),
        ("`orders`", "orders"),
        ('"orders"', "orders"),
        ("shop.orders", "shop.orders"),
        ("ob_orders$1", "ob_orders$1"),
    ],
)
def test_safe_identifier_accepts(name, expected):
    assert assert_safe_identifier(name) == expected


@pytest.mark.parametrize(
    "name",
    ["orders; drop table x", "orders where 1=1", "1orders", "", "a.b.c", "orders--x"],
)
def test_safe_identifier_rejects(name):
    with pytest.raises(ReadOnlyViolation):
        assert_safe_identifier(name)