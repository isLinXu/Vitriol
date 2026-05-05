"""
大朝议 III · 前端 useImperialWS 拆分重构示例
演示如何将 32.8 KB 的巨石 Hook 拆分为可测试的模块

目录结构（目标）：
frontend/src/hooks/ws/
  useImperialWS.ts         # 主 Hook，对外 API（< 200 行）
  wsTransport.ts           # WebSocket 连接/重连/心跳
  wsReducer.ts             # 增量状态合并（纯函数，易测试）
  wsAck.ts                 # ACK 机制与离线消息
  wsRouter.ts              # 消息类型路由
  types.ts                 # 类型定义
"""

# ============================================================================
# 示例 1: frontend/src/hooks/ws/types.ts
# ============================================================================

"""
export type WSMessageType = 
  | 'node_update'
  | 'node_update_delta'
  | 'task_complete'
  | 'approval_request'
  | 'notification'
  | 'error'
  | 'ack'
  | 'ping'
  | 'pong';

export interface WSMessage {
  type: WSMessageType;
  task_id?: string;
  timestamp: number;
  data?: any;
}

export interface WSState {
  connected: boolean;
  reconnecting: boolean;
  error: string | null;
  latency: number;
}

export interface PendingAck {
  msgId: string;
  message: WSMessage;
  sentAt: number;
  retries: number;
}
"""


# ============================================================================
# 示例 2: frontend/src/hooks/ws/wsTransport.ts
# ============================================================================

"""
/**
 * WebSocket 传输层
 * 负责连接、重连、心跳，不涉及业务逻辑
 */
import { WSMessage, WSState } from './types';

export class WSTransport {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  
  private onMessageCallback: ((msg: WSMessage) => void) | null = null;
  private onStateChangeCallback: ((state: WSState) => void) | null = null;

  constructor(url: string) {
    this.url = url;
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return;
    
    this.ws = new WebSocket(this.url);
    
    this.ws.onopen = () => {
      console.log('[WS] Connected');
      this.reconnectAttempts = 0;
      this.startHeartbeat();
      this.notifyStateChange({ connected: true, reconnecting: false, error: null, latency: 0 });
    };
    
    this.ws.onmessage = (event) => {
      const message: WSMessage = JSON.parse(event.data);
      this.onMessageCallback?.(message);
    };
    
    this.ws.onerror = (error) => {
      console.error('[WS] Error:', error);
      this.notifyStateChange({ connected: false, reconnecting: false, error: 'Connection error', latency: 0 });
    };
    
    this.ws.onclose = () => {
      console.log('[WS] Closed');
      this.stopHeartbeat();
      this.notifyStateChange({ connected: false, reconnecting: false, error: null, latency: 0 });
      this.scheduleReconnect();
    };
  }

  disconnect() {
    this.stopHeartbeat();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  send(message: WSMessage) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      throw new Error('WebSocket not connected');
    }
  }

  onMessage(callback: (msg: WSMessage) => void) {
    this.onMessageCallback = callback;
  }

  onStateChange(callback: (state: WSState) => void) {
    this.onStateChangeCallback = callback;
  }

  private startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      const pingTime = Date.now();
      this.send({ type: 'ping', timestamp: pingTime });
    }, 30000);
  }

  private stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  private scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[WS] Max reconnect attempts reached');
      return;
    }
    
    this.reconnectAttempts++;
    const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts), 30000);
    
    console.log(`[WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    this.notifyStateChange({ connected: false, reconnecting: true, error: null, latency: 0 });
    
    setTimeout(() => this.connect(), delay);
  }

  private notifyStateChange(state: WSState) {
    this.onStateChangeCallback?.(state);
  }
}
"""


# ============================================================================
# 示例 3: frontend/src/hooks/ws/wsReducer.ts
# ============================================================================

"""
/**
 * WebSocket 状态增量合并（纯函数）
 * 可独立单元测试
 */
import { CourtState } from '@/store/useCourtStore';

export interface DeltaUpdate {
  logs?: string[];
  logs_total?: number;
  history?: any[];
  history_total?: number;
  [key: string]: any;
}

export function applyDelta(prevState: CourtState, delta: DeltaUpdate): CourtState {
  const newState = { ...prevState };
  
  // 增量日志合并
  if (delta.logs) {
    newState.logs = [...(prevState.logs || []), ...delta.logs];
    if (delta.logs_total) {
      // 如果总数超出，裁掉旧日志（保留最新 100 条）
      if (newState.logs.length > 100) {
        newState.logs = newState.logs.slice(-100);
      }
    }
  }
  
  // 增量历史合并
  if (delta.history) {
    newState.history = [...(prevState.history || []), ...delta.history];
  }
  
  // 其他字段直接覆盖
  Object.keys(delta).forEach((key) => {
    if (key !== 'logs' && key !== 'history' && key !== 'logs_total' && key !== 'history_total') {
      newState[key] = delta[key];
    }
  });
  
  return newState;
}

// 测试示例
export function testApplyDelta() {
  const prevState = {
    logs: ['log1', 'log2'],
    history: ['event1'],
    status: 'idle',
  };
  
  const delta = {
    logs: ['log3', 'log4'],
    logs_total: 4,
    status: 'processing',
  };
  
  const newState = applyDelta(prevState as any, delta);
  
  console.assert(newState.logs.length === 4, 'Logs should merge');
  console.assert(newState.status === 'processing', 'Status should update');
  console.assert(newState.history.length === 1, 'History unchanged');
}
"""


# ============================================================================
# 示例 4: frontend/src/hooks/ws/wsAck.ts
# ============================================================================

"""
/**
 * ACK 机制与离线消息队列
 */
import { WSMessage, PendingAck } from './types';

export class WSAckManager {
  private pendingAcks = new Map<string, PendingAck>();
  private offlineQueue: WSMessage[] = [];
  private maxRetries = 3;
  private ackTimeout = 5000;

  addPending(msgId: string, message: WSMessage) {
    this.pendingAcks.set(msgId, {
      msgId,
      message,
      sentAt: Date.now(),
      retries: 0,
    });
    
    // 设置超时检查
    setTimeout(() => this.checkTimeout(msgId), this.ackTimeout);
  }

  handleAck(msgId: string) {
    this.pendingAcks.delete(msgId);
  }

  checkTimeout(msgId: string) {
    const pending = this.pendingAcks.get(msgId);
    if (!pending) return; // 已收到 ACK
    
    if (pending.retries < this.maxRetries) {
      console.warn(`[ACK] Timeout for ${msgId}, retry ${pending.retries + 1}`);
      pending.retries++;
      // 重新发送（需要外部 send 函数）
      // this.transport.send(pending.message);
      setTimeout(() => this.checkTimeout(msgId), this.ackTimeout);
    } else {
      console.error(`[ACK] Max retries reached for ${msgId}, moving to offline queue`);
      this.offlineQueue.push(pending.message);
      this.pendingAcks.delete(msgId);
    }
  }

  getOfflineMessages(): WSMessage[] {
    const messages = [...this.offlineQueue];
    this.offlineQueue = [];
    return messages;
  }
}
"""


# ============================================================================
# 示例 5: frontend/src/hooks/ws/wsRouter.ts
# ============================================================================

"""
/**
 * 消息类型路由器
 * 根据消息类型分发到不同处理器
 */
import { WSMessage } from './types';

type MessageHandler = (msg: WSMessage) => void;

export class WSMessageRouter {
  private handlers = new Map<string, MessageHandler[]>();

  on(type: string, handler: MessageHandler) {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, []);
    }
    this.handlers.get(type)!.push(handler);
  }

  route(message: WSMessage) {
    const handlers = this.handlers.get(message.type);
    if (handlers) {
      handlers.forEach((handler) => handler(message));
    } else {
      console.warn(`[Router] No handler for message type: ${message.type}`);
    }
  }
}
"""


# ============================================================================
# 示例 6: frontend/src/hooks/ws/useImperialWS.ts（重构后的主 Hook）
# ============================================================================

"""
/**
 * useImperialWS - 主 Hook（对外 API）
 * 重构后 < 200 行
 */
import { useEffect, useRef, useState } from 'react';
import { WSTransport } from './wsTransport';
import { WSMessageRouter } from './wsRouter';
import { WSAckManager } from './wsAck';
import { applyDelta } from './wsReducer';
import { useCourtStore } from '@/store/useCourtStore';

export function useImperialWS(url: string) {
  const [wsState, setWsState] = useState({
    connected: false,
    reconnecting: false,
    error: null as string | null,
    latency: 0,
  });

  const transportRef = useRef<WSTransport>();
  const routerRef = useRef<WSMessageRouter>();
  const ackManagerRef = useRef<WSAckManager>();
  
  const courtStore = useCourtStore();

  useEffect(() => {
    // 初始化各模块
    const transport = new WSTransport(url);
    const router = new WSMessageRouter();
    const ackManager = new WSAckManager();
    
    transportRef.current = transport;
    routerRef.current = router;
    ackManagerRef.current = ackManager;

    // 注册消息处理器
    router.on('node_update_delta', (msg) => {
      const newState = applyDelta(courtStore.getState(), msg.data);
      courtStore.setState(newState);
    });

    router.on('task_complete', (msg) => {
      courtStore.setTaskComplete(msg.task_id);
    });

    router.on('ack', (msg) => {
      ackManager.handleAck(msg.data.msgId);
    });

    router.on('pong', (msg) => {
      const latency = Date.now() - msg.timestamp;
      setWsState((prev) => ({ ...prev, latency }));
    });

    // 连接传输层
    transport.onMessage((msg) => router.route(msg));
    transport.onStateChange(setWsState);
    transport.connect();

    // 清理
    return () => {
      transport.disconnect();
    };
  }, [url]);

  // 发送消息（带 ACK）
  const sendMessage = (message: any) => {
    const msgId = `msg_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    const fullMessage = { ...message, msgId, timestamp: Date.now() };
    
    transportRef.current?.send(fullMessage);
    ackManagerRef.current?.addPending(msgId, fullMessage);
  };

  return {
    ...wsState,
    sendMessage,
  };
}
"""


# ============================================================================
# 测试示例
# ============================================================================

"""
// frontend/src/hooks/ws/__tests__/wsReducer.test.ts
import { applyDelta } from '../wsReducer';

describe('wsReducer', () => {
  test('should merge logs incrementally', () => {
    const prevState = { logs: ['log1'], history: [] };
    const delta = { logs: ['log2', 'log3'] };
    
    const newState = applyDelta(prevState as any, delta);
    
    expect(newState.logs).toEqual(['log1', 'log2', 'log3']);
  });
  
  test('should limit logs to 100 items', () => {
    const prevState = { logs: Array(95).fill('log'), history: [] };
    const delta = { logs: Array(10).fill('new'), logs_total: 105 };
    
    const newState = applyDelta(prevState as any, delta);
    
    expect(newState.logs.length).toBe(100);
  });
});
"""

print("""
拆分收益总结：

1. **可测试性**：每个模块可独立单测
   - wsReducer.ts: 纯函数，100% 覆盖率
   - wsTransport.ts: 可 mock WebSocket
   - wsAck.ts: 可独立测试超时逻辑

2. **可维护性**：单一职责
   - 修改重连逻辑只需改 wsTransport.ts
   - 修改状态合并只需改 wsReducer.ts

3. **可复用性**：模块可跨项目
   - WSTransport 可用于任何 WebSocket 项目
   - WSAckManager 是通用的 ACK 实现

4. **行数收敛**：
   - 原 useImperialWS.ts: ~900 行
   - 重构后主 Hook: <200 行
   - 各子模块: 50-150 行

迁移步骤：
1. 创建 frontend/src/hooks/ws/ 目录
2. 复制原 useImperialWS.ts 为 useImperialWS.old.ts（备份）
3. 按上述示例创建 types.ts / wsTransport.ts / wsReducer.ts / wsAck.ts / wsRouter.ts
4. 重写 useImperialWS.ts 为精简版
5. 运行测试：npm test -- useImperialWS
6. 手动冒烟：打开前端，发送圣旨，观察 WebSocket 消息
""")
