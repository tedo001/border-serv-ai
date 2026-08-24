"""MQTT sink.

The usual transport between a BOP node and a sector control room: lightweight,
tolerant of intermittent links, and already present in most command-and-control
estates. Topics are structured ``prefix/<camera>/<event_type>`` so a control
room can subscribe narrowly - one operator's console can take
``ibvap/events/+/intrusion`` without receiving every plate read on the sector.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ibvap.core.config import MqttConfig
from ibvap.core.errors import IntegrationError
from ibvap.core.logging import get_logger
from ibvap.integrations.base import Sink

log = get_logger(__name__)


class MqttSink(Sink):
    """Publishes events to an MQTT broker."""

    def __init__(self, config: MqttConfig, *, name: str = "mqtt") -> None:
        super().__init__(min_severity=config.min_severity, enabled=config.enabled)
        self.name = name
        self.config = config
        self._client: Any = None
        self._connected = False

    async def start(self) -> None:
        if not self.enabled or self._client is not None:
            return
        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise IntegrationError("paho-mqtt is not installed") from exc

        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"ibvap-{self.name}",
            protocol=mqtt.MQTTv5,
            # A clean session would silently discard queued QoS 1 messages
            # across a reconnect - exactly the alerts a flaky link must keep.
            clean_session=None,
        )
        if self.config.username:
            client.username_pw_set(self.config.username, self.config.password)
        if self.config.tls:
            client.tls_set()

        def _on_connect(_c: Any, _u: Any, _f: Any, reason: Any, _p: Any = None) -> None:
            self._connected = int(reason) == 0
            log.info("mqtt_connected" if self._connected else "mqtt_connect_failed",
                     sink=self.name, reason=str(reason))

        def _on_disconnect(_c: Any, _u: Any, _f: Any, reason: Any = None, _p: Any = None) -> None:
            self._connected = False
            log.warning("mqtt_disconnected", sink=self.name, reason=str(reason))

        client.on_connect = _on_connect
        client.on_disconnect = _on_disconnect
        # paho's own reconnect loop handles link flaps without us rebuilding
        # the client, which would drop its queued messages.
        client.reconnect_delay_set(min_delay=1, max_delay=60)

        try:
            await asyncio.to_thread(
                client.connect, self.config.host, self.config.port, 60
            )
        except OSError as exc:
            raise IntegrationError(
                f"mqtt sink {self.name} cannot reach {self.config.host}:{self.config.port}: {exc}"
            ) from exc

        client.loop_start()
        self._client = client

    async def close(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
            self._connected = False

    async def deliver(self, payload: dict[str, Any]) -> None:
        if self._client is None:
            await self.start()
        if self._client is None:
            raise IntegrationError(f"mqtt sink {self.name} is not connected")

        camera = payload.get("camera", {}).get("id", "unknown")
        event_type = payload.get("event_type", "unknown")
        topic = f"{self.config.topic_prefix}/{camera}/{event_type}"
        body = json.dumps(payload, separators=(",", ":"))

        info = self._client.publish(topic, body, qos=self.config.qos, retain=False)
        # Waiting for the broker acknowledgement is what makes QoS 1 meaningful
        # here: without it a "successful" publish only means the message
        # reached the local socket buffer, and the outbox would mark an
        # undelivered alert as delivered.
        if self.config.qos > 0:
            await asyncio.to_thread(info.wait_for_publish, 10.0)
        if info.rc != 0:
            raise IntegrationError(f"mqtt publish failed for {topic}: rc={info.rc}")

    def status(self) -> dict[str, Any]:
        return {
            **super().status(),
            "kind": "mqtt",
            "broker": f"{self.config.host}:{self.config.port}",
            "topic_prefix": self.config.topic_prefix,
            "connected": self._connected,
            "qos": self.config.qos,
        }
