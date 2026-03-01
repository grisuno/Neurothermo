#!/usr/bin/env python3
# _*_ coding: utf8 _*_
"""
app.py

Autor: Gris Iscomeback
Correo electrónico: grisiscomeback[at]gmail[dot]com
Fecha de creación: 28/02/2026
Licencia: GPL v3

Descripción:  QUICK START - neurothermo library.

During training: ONLY delta (instant, O(n))
At summary(): ALL 17 metrics computed from history
"""

import torch
import torch.nn as nn
from neurothermo import create_monitor

# Your model
model = nn.Linear(100, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ONE LINE
monitor = create_monitor(model)

# Training loop
for epoch in range(10):
    x = torch.randn(32, 100)
    y = torch.randn(32, 10)

    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()

    # ONE LINE - ONLY delta computed (instant)
    metrics = monitor.step(loss=loss.item())
    optimizer.step()

    print(f"Epoch {epoch}: Delta={metrics.get('delta'):.4f}")

# ALL metrics computed here
print()
print(monitor.summary())
