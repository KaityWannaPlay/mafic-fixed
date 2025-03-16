"""Node class to represent one Lavalink instance."""
# SPDX-License-Identifier: MIT
# pyright: reportImportCycles=false
# Player import.

from __future__ import annotations

import re
import warnings
from asyncio import Event, TimeoutError, create_task, gather, sleep, wait_for
from logging import getLogger
from traceback import print_exc
from typing import TYPE_CHECKING, Generic, cast

import aiohttp
import yarl

from .__libraries import MISSING, ExponentialBackoff, dumps, loads
from .errors import *
from .ip import (
    BalancingIPRoutePlannerStatus,
    NanoIPRoutePlannerStatus,
    RotatingIPRoutePlannerStatus,
    RotatingNanoIPRoutePlannerStatus,
)
from .playlist import Playlist
from .plugin import Plugin
from .region import Group, Region, VoiceRegion
from .stats import NodeStats
from .track import Track
from .type_variables import ClientT
from .typings import (
    BalancingIPRouteDetails,
    NanoIPRouteDetails,
    RotatingIPRouteDetails,
    RotatingNanoIPRouteDetails,
    TrackWithInfo,
)
from .warnings import *

if TYPE_CHECKING:
    from asyncio import Task
    from collections.abc import Sequence
    from typing import Any, Literal

    from aiohttp import ClientWebSocketResponse

    from .__libraries import VoiceServerUpdatePayload
    from .filter import Filter
    from .ip import RoutePlannerStatus
    from .player import Player
    from .typings import (
        Coro,
        EventPayload,
        IncomingMessage,
        OutgoingMessage,
        OutgoingParams,
        Player as PlayerPayload,
        PluginData,
        RoutePlannerStatus as RoutePlannerStatusPayload,
        TrackLoadingResult,
        UpdatePlayerParams,
        UpdatePlayerPayload,
        UpdateSessionPayload,
    )

_log = getLogger(__name__)
URL_REGEX = re.compile(r"https?://")


__all__ = ("Node",)


def _wrap_regions(
    regions: Sequence[Group | Region | VoiceRegion] | None,
) -> list[VoiceRegion] | None:
    r"""Convert a list of voice regions, regions and groups into a list of regions.

    Parameters
    ----------
    regions:
        The list of regions to convert.

    Returns
    -------
    :class:`list`\[:class:`Region`] | None
        The converted list of regions.
    """
    if not regions:
        return None

    actual_regions: list[VoiceRegion] = []

    for item in regions:
        if isinstance(item, Group):
            for region in item.value:
                actual_regions.extend(region.value)
        elif isinstance(item, Region):
            actual_regions.extend(item.value)
        elif isinstance(
            item, VoiceRegion
        ):  # pyright: ignore[reportUnnecessaryIsInstance]
            actual_regions.append(item)
        else:
            msg = f"Expected Group, Region, or VoiceRegion, got {type(item)!r}."
            raise TypeError(msg)

    return actual_regions


class Node(Generic[ClientT]):
    r"""Represents a Lavalink node.

    .. warning::

        This class should not be instantiated manually.
        Instead, use :meth:`NodePool.create_node`.

    Parameters
    ----------
    host:
        The host of the node, used to connect.
    port:
        The port of the node, used to connect.
    label:
        The label of the node, used to identify the node.
    password:
        The password of the node, used to authenticate the connection.
    client:
        The client that the node is attached to.
    secure:
        Whether the node is using a secure connection.
        This determines whether the node uses HTTP or HTTPS, WS or WSS.
    heartbeat:
        The interval at which the node will send a heartbeat to the client.
    timeout:
        The amount of time the node will wait for a response before raising a timeout
        error.
    session: :data:`~typing.Optional`\[:class:`aiohttp.ClientSession`]
        The session to use for the node.
        If not provided, a new session will be created.
    resume_key:
        The key to use when resuming the node.
        If not provided, the key will be generated from the host, port and label.

        .. warning::

            This is ignored in lavalink V4, use ``resuming_session_id`` instead.
    regions:
        The voice regions that the node can be used in.
        This is used to determine when to use this node.
    shard_ids:
        The shard IDs that the node can be used in.
        This is used to determine when to use this node.
    resuming_session_id:
        The session ID to use when resuming the node.
        If not provided, the node will not resume.

        This should be stored from :func:`~mafic.on_node_ready` with :attr:`session_id`
        to resume the session and gain control of the players. If the node is not
        resuming, players will be destroyed if Lavalink loses connection to us.

        .. versionadded:: 2.2

    Attributes
    ----------
    regions: :data:`~typing.Optional`\[:class:`list`\[:class:`~.VoiceRegion`]]
        The regions that the node can be used in.
        This is used to determine when to use this node.
    shard_ids: :data:`~typing.Optional`\[:class:`list`\[:class:`int`]]
        The shard IDs that the node can be used in.
        This is used to determine when to use this node.
    """

    __slots__ = (
        "__password",
        "__session",
        "_available",
        "_checked_version",
        "_client",
        "_connect_task",
        "_heartbeat",
        "_host",
        "_label",
        "_msg_tasks",
        "_players",
        "_port",
        "_resume_key",
        "_secure",
        "_timeout",
        "_ready",
        "_rest_uri",
        "_resuming_session_id",
        "_session_id",
        "_stats",
        "_version",
        "_ws",
        "_ws_uri",
        "_ws_task",
        "_event_queue",
        "regions",
        "shard_ids",
    )

    def __init__(
        self,
        *,
        host: str,
        port: int,
        label: str,
        password: str,
        client: ClientT,
        secure: bool = False,
        heartbeat: int = 30,
        timeout: float = 10,
        session: aiohttp.ClientSession | None = None,
        resume_key: str | None = None,
        regions: Sequence[Group | Region | VoiceRegion] | None = None,
        shard_ids: Sequence[int] | None = None,
        resuming_session_id: str | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._label = label
        self.__password = password
        self._secure = secure
        self._heartbeat = heartbeat
        self._timeout = timeout
        self._client = client
        self.__session = session
        self.shard_ids: Sequence[int] | None = shard_ids
        self.regions: list[VoiceRegion] | None = _wrap_regions(regions)

        self._rest_uri = yarl.URL.build(
            scheme=f"http{'s'*secure}", host=host, port=port
        )
        self._ws_uri = yarl.URL.build(scheme=f"ws{'s'*secure}", host=host, port=port)
        self._resume_key = resume_key or f"{host}:{port}:{label}"
        self._resuming_session_id: str = resuming_session_id or ""

        self._ws: ClientWebSocketResponse | None = None
        self._ws_task: Task[None] | None = None

        self._available = False
        self._ready = Event()

        self._players: dict[int, Player[ClientT]] = {}

        self._stats: NodeStats | None = None
        self._session_id: str | None = None

        self._msg_tasks: set[Task[None]] = set()
        self._connect_task: Task[None] | None = None

        self._checked_version: bool = False
        self._version: int = 3

        self._event_queue: Event = Event()

    @property
    def host(self) -> str:
        """The host of the node."""
        return self._host

    @property
    def port(self) -> int:
        """The port of the node."""
        return self._port

    @property
    def label(self) -> str:
        """The label of the node."""
        return self._label

    @property
    def client(self) -> ClientT:
        """The client that the node is attached to."""
        return self._client

    @property
    def secure(self) -> bool:
        """Whether the node is using a secure connection."""
        return self._secure

    @property
    def stats(self) -> NodeStats | None:
        """The stats of the node.

        This will be ``None`` if the node has not sent stats yet.
        This could be if it is not connected, or if stats sending is disabled on the
        node.
        """
        return self._stats

    @property
    def available(self) -> bool:
        """Whether the node is available.

        This is ``False`` if the node is not connected, or if it is not ready.
        """
        return self._available

    @property
    def weight(self) -> float:
        """The weight of the node.

        This is used to determine which node to use when multiple nodes are available.

        Notes
        -----
        The weight is calculated based on the following:

        - The number of players connected to the node.
        - The load of the node.
        - The number of UDP frames nulled.
        - The number of UDP frames that are lost.
        - If the node memory is very close to full.

        If the node has not sent stats yet, then a high value will be returned.
        This is so that the node will be used if it is the only one available,
        or if stats sending is disabled on the node.
        """
        if self._stats is None:
            # Stats haven't been set yet, so we'll just return a high value.
            # This is so we can properly balance known nodes.
            # If stats sending is turned off
            # - that's on the user
            # - they likely have done it on all if they have multiple, so it is equal
            return 6.63e34

        stats = self._stats

        players = stats.playing_player_count

        # These are exponential equations.

        # Load is *basically* a percentage (I know it isn't but it is close enough).

        # | cores | load | weight |
        # |-------|------|--------|
        # | 1     | 0.1  | 16     |
        # | 1     | 0.5  | 114    |
        # | 1     | 0.75 | 388    |
        # | 1     | 1    | 1315   |
        # | 3     | 0.1  | 12     |
        # | 3     | 1    | 51     |
        # | 3     | 2    | 259    |
        # | 3     | 3    | 1315   |
        cpu = 1.05 ** (100 * (stats.cpu.system_load / stats.cpu.cores)) * 10 - 10

        # | null frames | weight |
        # | ----------- | ------ |
        # | 10          | 30     |
        # | 20          | 62     |
        # | 100         | 382    |
        # | 250         | 1456   |

        frame_stats = stats.frame_stats
        if frame_stats is None:
            null = 0
            deficit = 0
        else:
            null = 1.03 ** (frame_stats.nulled / 6) * 600 - 600
            deficit = 1.03 ** (frame_stats.deficit / 6) * 600 - 600

        # High memory usage isnt bad, but we generally don't want to overload it.
        # Especially due to the chance of regular GC pauses.

        # | memory usage | weight |
        # | ------------ | ------ |
        # | 96%          | 0      |
        # | 97%          | 9      |
        # | 98%          | 99     |
        # | 99%          | 999    |
        # | 99.5%        | 3161   |
        # | 100%         | 9999   |

        mem_stats = stats.memory
        mem = max(10 ** (100 * (mem_stats.used / mem_stats.reservable) - 96), 1) - 1

        return players + cpu + null + deficit + mem

    @property
    def players(self) -> list[Player[ClientT]]:
        """The players connected to the node.

        .. versionchanged:: 2.0

            This is now a list.
        """
        return [*self._players.values()]

    @property
    def session_id(self) -> str | None:
        """The session ID of the node.

        This is ``None`` if the node is not connected.

        .. versionadded:: 2.2
        """
        return self._session_id

    @property
    def version(self) -> int:
        """The major semver version of the node.

        This is ``3`` if the node is not connected.
        This is mostly used in :class:`Player` for version checks.

        .. versionadded:: 2.2
        """
        return self._version

    def get_player(self, guild_id: int) -> Player[ClientT] | None:
        r"""Get a player from the node.

        Parameters
        ----------
        guild_id:
            The guild ID to get the player for.

        Returns
        -------
        :data:`~typing.Optional`\[:class:`Player`]
            The player for the guild, if found.
        """
        return self._players.get(guild_id)

    async def fetch_player(self, guild_id: int) -> PlayerPayload:
        """Fetch player data from the node.

        .. note::

            This is an API call. Usually you should use :meth:`get_player` instead.

        .. versionadded:: 2.6

        Parameters
        ----------
        guild_id:
            The guild ID to fetch the player for.

        Returns
        -------
        :class:`dict`
            The player data for the guild.
        """
        return await self.__request(
            "GET", f"sessions/{self._session_id}/players/{guild_id}"
        )

    def add_player(self, guild_id: int, player: Player[ClientT]) -> None:
        """Add a player to the node.

        Parameters
        ----------
        guild_id:
            The guild ID to add the player for.
        player:
            The player to add.
        """
        self._players[guild_id] = player

    def remove_player(self, guild_id: int) -> None:
        """Remove a player from the node.

        .. note::

            This does not disconnect the player from the voice channel.
            This simply exists to remove the player from the node cache.

        Parameters
        ----------
        guild_id:
            The guild ID to remove the player for.
        """
        self._players.pop(guild_id, None)

    async def _check_version(self) -> int:
        """:class:`int`: The major version of the node.

        This also does checks based on if that is supported.

        Raises
        ------
        :exc:`RuntimeError`
            If the
            - major version is not in (3, 4)
            - minor version is less than 7 when the major version is 3

            This is because the rest api is in 3.7, and v5 will have breaking changes.

        Warns
        -----
        :class:`UnsupportedVersionWarning`
            If the
            - major version is 3 and the minor version is more than 7
            - major version is 4 and the minor version is more than 0
            Some features may not work.
        """
        if self._checked_version:
            # This process was already ran likely.
            return self._version

        if self.__session is None:
            self.__session = await self._create_session()

        async with self.__session.get(
            self._rest_uri / "version",
            headers={"Authorization": self.__password},
        ) as resp:
            # Only the major and minor are needed.
            version = await resp.text()

            try:
                major, minor, _ = version.split(".", maxsplit=2)
            except ValueError:
                if version.endswith("-SNAPSHOT"):
                    major = 4
                    minor = 0
                else:
                    major = 3
                    minor = 7
                    message = UnknownVersionWarning.message
                    warnings.warn(message, UnknownVersionWarning, stacklevel=4)
            else:
                major = int(major)
                minor = int(minor)

                if major not in (3, 4) or (major == 3 and minor < 7):
                    msg = (
                        f"Unsupported lavalink version {version} "
                        "(expected 3.7.x or 4.x.x)"
                    )
                    raise RuntimeError(msg)
                elif (major == 3 and minor > 7) or (major == 4 and minor > 0):
                    message = UnsupportedVersionWarning.message
                    warnings.warn(message, UnsupportedVersionWarning, stacklevel=4)

            self._rest_uri /= f"v{major}"
            self._ws_uri /= f"v{major}/websocket"

            self._version = major
            self._checked_version = True
            return major

    async def _connect_to_websocket(
        self, headers: dict[str, str], session: aiohttp.ClientSession
    ) -> None:
        """Connect to the websocket of the node.

        Parameters
        ----------
        headers:
            The headers to use for the websocket connection.
        session:
            The session to use for the websocket connection.
        """
        try:
            self._ws = (
                await session.ws_connect(  # pyright: ignore[reportUnknownMemberType]
                    self._ws_uri,
                    timeout=self._timeout,
                    heartbeat=self._heartbeat,
                    headers=headers,
                )
            )
        except Exception as e:
            _log.error(
                "Failed to connect to lavalink at %s: %s",
                self._rest_uri,
                e,
                extra={"label": self._label},
            )
            raise

    async def connect(
        self,
        *,
        backoff: ExponentialBackoff[Literal[False]] | None = None,
        player_cls: type[Player[ClientT]] | None = None,
    ) -> None:
        """Connect to the node.

        Parameters
        ----------
        backoff:
            The backoff to use when reconnecting.
        player_cls:
            The player class to use for the node when resuming.

            .. versionadded:: 2.8

        Raises
        ------
        NodeAlreadyConnected
            If the node is already connected.
        asyncio.TimeoutError
            If the connection times out.
            You can change the timeout with the `timeout` parameter.
        """
        if self._ws is not None:
            raise NodeAlreadyConnected

        _log.info("Waiting for client to be ready...", extra={"label": self._label})
        await self._client.wait_until_ready()
        if self._client.user is None:
            msg = "Client.user is None"
            raise RuntimeError(msg)

        if self.__session is None:
            self.__session = await self._create_session()

        session = self.__session

        _log.debug("Checking lavalink version...", extra={"label": self._label})
        version = await self._check_version()

        headers: dict[str, str] = {
            "Authorization": self.__password,
            "User-Id": str(self._client.user.id),
            "Client-Name": f"Mafic/{__import__('mafic').__version__}",
        }

        # V4 uses session ID resuming
        if version == 3:
            headers["Resume-Key"] = self._resume_key
        else:
            headers["Session-Id"] = self._resuming_session_id

        _log.info(
            "Connecting to lavalink at %s...",
            self._rest_uri,
            extra={"label": self._label},
        )
        try:
            await self._connect_to_websocket(headers=headers, session=session)
        except Exception as e:  # noqa: BLE001
            _log.error(
                "Failed to connect to lavalink at %s: %s",
                self._rest_uri,
                e,
                extra={"label": self._label},
            )
            print_exc()

            backoff = backoff or ExponentialBackoff()
            delay = backoff.delay()
            _log.info(
                "Retrying connection to lavalink at %s in %s seconds...",
                self._rest_uri,
                delay,
                extra={"label": self._label},
            )
            await sleep(delay)

            task = create_task(self.connect(backoff=backoff))
            self._connect_task = task

            def remove_ta
