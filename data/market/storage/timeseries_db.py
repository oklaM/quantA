"""
时序数据库封装
支持DuckDB和ClickHouse
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import pandas as pd
from pathlib import Path

from config.settings import database as db_config
from utils.logging import get_logger

logger = get_logger(__name__)


class BaseTimeSeriesDB(ABC):
    """时序数据库基类"""

    def __init__(self):
        self._is_connected = False

    @abstractmethod
    def connect(self):
        """建立连接"""
        pass

    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass

    @abstractmethod
    def write(self, table_name: str, data: pd.DataFrame):
        """写入数据"""
        pass

    @abstractmethod
    def read(
        self,
        table_name: str,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """读取数据"""
        pass

    @abstractmethod
    def create_table(self, table_name: str, schema: Dict[str, str]):
        """创建表"""
        pass

    @abstractmethod
    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        pass

    @property
    def is_connected(self) -> bool:
        return self._is_connected


class DuckDBTimeSeries(BaseTimeSeriesDB):
    """DuckDB时序数据库实现"""

    def __init__(self, db_path: Optional[Path] = None):
        super().__init__()
        self.db_path = db_path or db_config.DUCKDB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None

    def connect(self):
        """建立连接"""
        try:
            import duckdb
        except ImportError:
            raise ImportError("duckdb未安装，请运行: pip install duckdb")

        self._conn = duckdb.connect(str(self.db_path))
        self._is_connected = True
        logger.info(f"DuckDB连接成功: {self.db_path}")
        return self

    def disconnect(self):
        """断开连接"""
        if self._conn:
            self._conn.close()
        self._is_connected = False
        logger.info("DuckDB已断开")

    def write(self, table_name: str, data: pd.DataFrame):
        """写入数据"""
        if not self.is_connected:
            self.connect()

        try:
            # 使用INSERT OR REPLACE避免重复
            self._conn.register("temp_data", data)
            self._conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM temp_data WHERE 1=0;
            """)

            # 获取主键列
            columns = self._conn.execute(f"PRAGMA table_info({table_name})").df()
            primary_keys = columns[columns["pk"] > 0]["name"].tolist()

            if primary_keys:
                # 有主键，使用INSERT OR REPLACE
                self._conn.execute(f"INSERT OR REPLACE INTO {table_name} SELECT * FROM temp_data")
            else:
                # 无主键，直接追加
                self._conn.execute(f"INSERT INTO {table_name} SELECT * FROM temp_data")

            self._conn.unregister("temp_data")
            logger.debug(f"写入{len(data)}条数据到表 {table_name}")

        except Exception as e:
            logger.error(f"写入数据失败: {e}")
            raise

    def read(
        self,
        table_name: str,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """读取数据"""
        if not self.is_connected:
            self.connect()

        if not self.table_exists(table_name):
            logger.warning(f"表 {table_name} 不存在")
            return pd.DataFrame()

        # 构建查询
        query = f"SELECT * FROM {table_name}"
        conditions = []

        if symbol:
            conditions.append(f"symbol = '{symbol}'")

        if start_date:
            # 尝试多种日期列名
            date_col = self._get_date_column(table_name)
            conditions.append(f"{date_col} >= '{start_date}'")

        if end_date:
            date_col = self._get_date_column(table_name)
            conditions.append(f"{date_col} <= '{end_date}'")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY date"

        try:
            return self._conn.execute(query).df()
        except Exception as e:
            logger.error(f"读取数据失败: {e}")
            return pd.DataFrame()

    def create_table(self, table_name: str, schema: Dict[str, str]):
        """创建表"""
        if not self.is_connected:
            self.connect()

        # 构建CREATE TABLE语句
        columns = []
        for col_name, col_type in schema.items():
            col_def = f"{col_name} {col_type}"
            if col_name == "symbol":
                col_def += " PRIMARY KEY"
            columns.append(col_def)

        create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"

        try:
            self._conn.execute(create_sql)
            logger.info(f"创建表 {table_name}")
        except Exception as e:
            logger.error(f"创建表失败: {e}")
            raise

    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        if not self.is_connected:
            self.connect()

        try:
            result = self._conn.execute(
                f"SELECT table_name FROM information_schema.tables WHERE table_name = '{table_name}'"
            ).fetchone()
            return result is not None
        except Exception:
            return False

    def _get_date_column(self, table_name: str) -> str:
        """获取日期列名"""
        # 尝试常见的日期列名
        for col in ["date", "time", "trade_date", "datetime"]:
            try:
                result = self._conn.execute(
                    f"PRAGMA table_info({table_name})"
                ).df()
                if col in result["name"].values:
                    return col
            except Exception:
                continue
        return "date"

    def execute_sql(self, sql: str) -> pd.DataFrame:
        """执行SQL查询"""
        if not self.is_connected:
            self.connect()

        try:
            return self._conn.execute(sql).df()
        except Exception as e:
            logger.error(f"执行SQL失败: {e}")
            raise


class ClickHouseTimeSeries(BaseTimeSeriesDB):
    """ClickHouse时序数据库实现（可选）"""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        super().__init__()
        self.host = host or db_config.CLICKHOUSE_HOST
        self.port = port or db_config.CLICKHOUSE_PORT
        self.user = user or db_config.CLICKHOUSE_USER
        self.password = password or db_config.CLICKHOUSE_PASSWORD
        self.database = database or db_config.CLICKHOUSE_DATABASE
        self._client = None

    def connect(self):
        """建立连接"""
        try:
            import clickhouse_connect
        except ImportError:
            raise ImportError("clickhouse_connect未安装，请运行: pip install clickhouse-connect")

        self._client = clickhouse_connect.get_client(
            host=self.host,
            port=self.port,
            username=self.user,
            password=self.password,
            database=self.database,
        )
        self._is_connected = True
        logger.info(f"ClickHouse连接成功: {self.host}:{self.port}")
        return self

    def disconnect(self):
        """断开连接"""
        if self._client:
            self._client.close()
        self._is_connected = False
        logger.info("ClickHouse已断开")

    def write(self, table_name: str, data: pd.DataFrame):
        """写入数据"""
        if not self.is_connected:
            self.connect()

        try:
            self._client.insert_df(table_name, data)
            logger.debug(f"写入{len(data)}条数据到表 {table_name}")
        except Exception as e:
            logger.error(f"写入数据失败: {e}")
            raise

    def read(
        self,
        table_name: str,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """读取数据"""
        if not self.is_connected:
            self.connect()

        # 构建查询
        query = f"SELECT * FROM {table_name}"
        conditions = []

        if symbol:
            conditions.append(f"symbol = '{symbol}'")

        if start_date:
            conditions.append(f"date >= '{start_date}'")

        if end_date:
            conditions.append(f"date <= '{end_date}'")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY date"

        try:
            return self._client.query_df(query)
        except Exception as e:
            logger.error(f"读取数据失败: {e}")
            return pd.DataFrame()

    def create_table(self, table_name: str, schema: Dict[str, str]):
        """创建表"""
        if not self.is_connected:
            self.connect()

        # 构建CREATE TABLE语句
        columns = []
        for col_name, col_type in schema.items():
            # 映射类型到ClickHouse
            ch_type = self._map_type_to_clickhouse(col_type)
            columns.append(f"{col_name} {ch_type}")

        create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)}) ENGINE = MergeTree() ORDER BY (symbol, date)"

        try:
            self._client.command(create_sql)
            logger.info(f"创建表 {table_name}")
        except Exception as e:
            logger.error(f"创建表失败: {e}")
            raise

    def table_exists(self, table_name: str) -> bool:
        """检查表是否存在"""
        if not self.is_connected:
            self.connect()

        try:
            result = self._client.query(f"EXISTS TABLE {table_name}")
            return result.first_row == 1
        except Exception:
            return False

    def _map_type_to_clickhouse(self, python_type: str) -> str:
        """将Python类型映射到ClickHouse类型"""
        type_map = {
            "VARCHAR": "String",
            "TEXT": "String",
            "INTEGER": "Int32",
            "BIGINT": "Int64",
            "FLOAT": "Float32",
            "DOUBLE": "Float64",
            "DATE": "Date",
            "DATETIME": "DateTime",
            "TIMESTAMP": "DateTime",
        }
        return type_map.get(python_type.upper(), "String")


# 工厂函数
def get_timeseries_db() -> BaseTimeSeriesDB:
    """获取时序数据库实例"""
    db_type = db_config.TIMESERIES_DB.lower()

    if db_type == "duckdb":
        return DuckDBTimeSeries()
    elif db_type == "clickhouse":
        return ClickHouseTimeSeries()
    else:
        raise ValueError(f"不支持的数据库类型: {db_type}")


# 预定义的表Schema
TABLE_SCHEMAS = {
    "daily_bar": {
        "symbol": "VARCHAR",
        "date": "DATE",
        "open": "DOUBLE",
        "high": "DOUBLE",
        "low": "DOUBLE",
        "close": "DOUBLE",
        "volume": "BIGINT",
        "amount": "DOUBLE",
    },
    "minute_bar": {
        "symbol": "VARCHAR",
        "time": "DATETIME",
        "open": "DOUBLE",
        "high": "DOUBLE",
        "low": "DOUBLE",
        "close": "DOUBLE",
        "volume": "BIGINT",
        "amount": "DOUBLE",
    },
    "stock_info": {
        "symbol": "VARCHAR PRIMARY KEY",
        "name": "VARCHAR",
        "market": "VARCHAR",
        "industry": "VARCHAR",
        "list_date": "DATE",
    },
}


__all__ = [
    'BaseTimeSeriesDB',
    'DuckDBTimeSeries',
    'ClickHouseTimeSeries',
    'get_timeseries_db',
    'TABLE_SCHEMAS',
]
