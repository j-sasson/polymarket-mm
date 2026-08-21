"""Builds an authenticated py-clob-client ClobClient from environment
variables, following https://docs.polymarket.com/trading/wallets-auth.

SECURITY: this module reads a private key from the environment and never
writes it, or the derived API secret/passphrase, to any log, print, or
exception message. It never accepts a private key as a function argument,
CLI flag, or config file value -- only via POLYMARKET_PRIVATE_KEY in the
environment, so it never ends up in shell history, a config file committed
by accident, or a log line.

Nothing in this module submits an order. It only establishes credentials.
"""
from __future__ import annotations

import os

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds

CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet -- Polymarket has no meaningful testnet

# signatureType per docs.polymarket.com/trading/place-orders: 0=EOA, 1=Proxy,
# 2=Safe, 3=Deposit. Deposit Wallet is the current default for new accounts.
DEFAULT_SIGNATURE_TYPE = 3


class MissingCredentialsError(RuntimeError):
    pass


def build_client(signature_type: int | None = None) -> ClobClient:
    """Level 1 client: can sign orders and derive/create L2 API creds, but
    cannot yet submit them (see `authenticate` for that)."""
    private_key = os.environ.get("POLYMARKET_PRIVATE_KEY")
    if not private_key:
        raise MissingCredentialsError(
            "POLYMARKET_PRIVATE_KEY is not set. Set it in your shell environment "
            "(never pass it as a CLI argument or hardcode it) before running live."
        )
    funder = os.environ.get("POLYMARKET_FUNDER_ADDRESS")  # required for Proxy/Safe/Deposit wallets
    sig_type = signature_type if signature_type is not None else int(
        os.environ.get("POLYMARKET_SIGNATURE_TYPE", DEFAULT_SIGNATURE_TYPE)
    )
    return ClobClient(
        CLOB_HOST,
        chain_id=CHAIN_ID,
        key=private_key,
        signature_type=sig_type,
        funder=funder,
    )


def authenticate(client: ClobClient) -> ClobClient:
    """Level 2: derives (or creates, on first use) API credentials via the
    L1-signed EIP-712 ClobAuth flow, then attaches them to the client so it
    can place/cancel orders. Credentials are held only in process memory."""
    creds: ApiCreds = client.create_or_derive_api_creds()
    client.set_api_creds(creds)
    return client
