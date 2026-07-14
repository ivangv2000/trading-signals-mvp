# Vercel wrapper — V14 Trading Signals

Este directorio es un **envoltorio estático** para mostrar la app Streamlit bajo tu dominio en Vercel. No ejecuta Python ni duplica el algoritmo V14.

## Arquitectura

1. **Streamlit Community Cloud** ejecuta `app.py` (motor real).
2. **Vercel** sirve `index.html`, que embebe la app con un iframe.

```
Usuario → tu-dominio.com (Vercel) → iframe → STREAMLIT_APP_URL/?embed=true
```

## Configuración

1. Despliega el repositorio en [Streamlit Community Cloud](https://share.streamlit.io/):
   - **Main file:** `app.py`
   - **Requirements:** `requirements.txt` (raíz del repo)

2. Copia la URL pública de Streamlit (ej. `https://tu-app.streamlit.app`).

3. En `index.html`, sustituye el placeholder:

   ```
   STREAMLIT_APP_URL
   ```

   por la URL real **sin barra final**, por ejemplo:

   ```html
   src="https://tu-app.streamlit.app/?embed=true"
   ```

4. En [Vercel](https://vercel.com/), crea un proyecto:
   - **Root Directory:** `vercel_site`
   - Framework: Other / Static
   - No requiere build command.

5. Asigna tu dominio personalizado en Vercel → Settings → Domains.

## Comprobación local

Abre `index.html` en el navegador tras reemplazar `STREAMLIT_APP_URL`. Debe ocupar el 100% de la pantalla con fondo oscuro `#0b1220`.

## Seguridad

- No subas `.env` ni `.streamlit/secrets.toml`.
- La app es paper trading; `APPROVED_FOR_REAL_MONEY=False`.
