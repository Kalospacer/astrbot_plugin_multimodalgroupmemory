# GroupMemory 插件

用于接管 AstrBot 群聊上下文记忆，解决“先发图后唤醒”场景下图片丢失的问题。

## 功能

- 记录群聊结构化消息（文本 / At / 图片）
- 首见图片：原生注入到 `req.image_urls`
- 复见图片：文本化为 `[Image]` 或 `[Image: xxx]`
- 支持 pending 队列与等待轮次淘汰
- 兼容 `/reset` / `/new` 的 `_clean_ltm_session` 清理

## 关键配置

- `enable_group_context`：是否启用接管（默认 `true`）
- `group_message_max_cnt`：最大会话消息数（默认 `300`）
- `history_message_window`：候选图片扫描窗口（默认 `5`）
- `max_native_images_per_round`：每轮注入上限（默认 `2`）
- `pending_max_wait_rounds`：pending 最大等待轮次（默认 `2`）
- `image_caption` / `image_caption_provider_id`：复见图片是否转述

## 注意

建议关闭 AstrBot 内置 LTM：

- `provider_ltm_settings.group_icl_enable = false`
- `provider_ltm_settings.active_reply.enable = false`

如果未关闭，插件会输出告警但不会阻断运行。
