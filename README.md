# ✦ Modular RAG Engine

> An enterprise-grade, extensible Retrieval-Augmented Generation (RAG) architecture built on SOLID principles.

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)
![Architecture](https://img.shields.io/badge/architecture-SOLID%20%7C%20Plug--and--Play-success.svg?style=flat-square)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=flat-square)

The **Modular RAG Engine** is a highly decoupled, schema-driven pipeline designed for clinical, legal, and enterprise AI applications. It leverages the **Open/Closed Principle (OCP)** and a **Dynamic Component Registry** to allow seamless integration of new parsers, chunkers, and embedding models without altering core system logic.

---

## Core Philosophy

- **Strict Data Contracts:** Immutable schemas (`SourceMetadata`) guarantee evidence traceability and auditability.
- **Dependency Inversion:** Execution pipelines depend entirely on Abstract Base Classes (ABCs), ensuring zero hard-coupling to external libraries.
- **Plug-and-Play Extensibility:** A Singleton Registry binds file extensions and configuration keys to their concrete implementations dynamically at runtime.
- **Configuration-Driven:** Pipeline behavior is fully controlled via YAML, requiring no code changes to swap components.

---

