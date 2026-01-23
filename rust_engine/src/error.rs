/*!
错误类型定义
*/

use thiserror::Error;

/// 回测引擎错误类型
#[derive(Error, Debug)]
pub enum EngineError {
    /// 数据错误
    #[error("数据错误: {0}")]
    DataError(String),

    /// 订单错误
    #[error("订单错误: {0}")]
    OrderError(String),

    /// 持仓错误
    #[error("持仓错误: {0}")]
    PositionError(String),

    /// 计算错误
    #[error("计算错误: {0}")]
    ComputationError(String),

    /// 配置错误
    #[error("配置错误: {0}")]
    ConfigError(String),

    /// IO错误
    #[error("IO错误: {0}")]
    IoError(#[from] std::io::Error),

    /// 序列化错误
    #[error("序列化错误: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// 数值错误
    #[error("数值错误: {0}")]
    NumericError(String),

    /// 时间错误
    #[error("时间错误: {0}")]
    TimeError(String),

    /// 其他错误
    #[error("未知错误: {0}")]
    Other(String),
}

/// 引擎结果类型
pub type EngineResult<T> = Result<T, EngineError>;

/// 从None转换错误
pub trait OptionExt<T> {
    fn or_else_err<F>(self, f: F) -> EngineResult<T>
    where
        F: FnOnce() -> String;
}

impl<T> OptionExt<T> for Option<T> {
    fn or_else_err<F>(self, f: F) -> EngineResult<T>
    where
        F: FnOnce() -> String,
    {
        self.ok_or_else(|| EngineError::Other(f()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = EngineError::DataError("测试数据错误".to_string());
        assert!(err.to_string().contains("数据错误"));
    }

    #[test]
    fn test_option_ext() {
        let some_value: Option<i32> = Some(42);
        assert_eq!(some_value.or_else_err(|| "错误".to_string()).unwrap(), 42);

        let none_value: Option<i32> = None;
        assert!(none_value.or_else_err(|| "值为空".to_string()).is_err());
    }
}
