# Vercel wrapper — V14 Trading Signals

Envoltorio estático para mostrar la app Streamlit bajo tu dominio en Vercel. No ejecuta Python ni duplica el algoritmo V14.

## Arquitectura

```
Usuario → tu-dominio.com (Vercel) → iframe → PEGA_AQUI_MI_URL_DE_STREAMLIT/?embed=true
```

## Configuración

1. Despliega la app en [Streamlit Community Cloud](https://share.streamlit.io/) con `app.py`.
2. En `index.html`, sustituye `PEGA_AQUI_MI_URL_DE_STREAMLIT` por tu URL real **sin barra final**:

   ```html
   src="https://tu-app.streamlit.app/?embed=true"
   ```

3. En [Vercel](https://vercel.com/):
   - **Root Directory:** `vercel_site`
   - Framework: Other / Static
   - Sin build command

4. Asigna tu dominio en Vercel → Settings → Domains.

## Comprobación

El iframe debe ocupar el 100% de ancho y alto, fondo `#0b1220`, sin scroll exterior, y mostrar «Cargando V14 Trading Signals…» al cargar.
