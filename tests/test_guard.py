import pytest

from app.tools.sql.guard import ReadOnlyViolation, assert_read_only

ALLOWED = [
    "select * from orders",
    "SELECT id FROM orders WHERE status='PAID'",
    "  -- comment\nselect 1",
    "show tables",
    "SHOW CREATE TABLE orders",
    "explain select * from orders",
    "describe orders",
    "desc orders",
    "with x as (select 1) select * from x",
    "select * from orders;",  # 尾部单分号允许
    "/* multi\n line */ select 1",
    "# hi\nselect 1",
]

DENIED = [
    "update orders set status='x'",
    "DELETE FROM orders",
    "insert into orders values (1)",
    "drop table orders",
    "alter table orders add column c int",
    "truncate table orders",
    "select * from orders for update",
    "select * from orders; drop table orders",  # 多语句
    "call some_proc()",
    "create index idx on orders(id)",
    "select * from orders for /* x */ update",
    "select * from orders lock /* x */ in share mode",
    "select * from orders into outfile '/tmp/x'",
    "select 'a' into dumpfile '/tmp/y'",
    "SELECT 1 INTO OUTFILE '/var/lib/mysql/x'",  # 大小写变体，验证 re.I 生效
    "SELECT 1 INTO /* c */ OUTFILE '/tmp/f'",  # 块注释变体，验证注释容忍
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_allows_readonly(sql):
    assert_read_only(sql)  # 不应抛异常


@pytest.mark.parametrize("sql", DENIED)
def test_denies_writes(sql):
    with pytest.raises(ReadOnlyViolation):
        assert_read_only(sql)
