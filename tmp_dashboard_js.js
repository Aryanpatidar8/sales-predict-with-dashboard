document.addEventListener('DOMContentLoaded', () => {
  const chartCtx = document.getElementById('predChart').getContext('2d')
  const tableBody = document.querySelector('#historyTable tbody')

  // Create Chart.js chart only if Chart is available; otherwise we'll fall back
  let chart = null
  if (typeof Chart !== 'undefined') {
    chart = new Chart(chartCtx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Prediction',
          data: [],
          borderColor: 'rgba(110,231,183,0.9)',
          backgroundColor: 'rgba(110,231,183,0.08)'
        }]
      },
      options: {
        responsive: true,
        scales: {
          x: { display: true },
          y: { display: true }
        }
      }
    })
  }

  function renderFallbackSVG(labels, values){
    const wrap = document.querySelector('.chart-wrap')
    // remove existing fallback
    const existing = wrap.querySelector('svg.fallback-chart')
    if(existing) existing.remove()
    const svgNS = 'http://www.w3.org/2000/svg'
    const w = wrap.clientWidth || 600
    const h = 280
    const svg = document.createElementNS(svgNS, 'svg')
    svg.setAttribute('width', w)
    svg.setAttribute('height', h)
    svg.classList.add('fallback-chart')
    // draw axes
    const padding = 30
    const max = Math.max(...values, 1)
    const min = Math.min(...values, 0)
    const range = max - min || 1
    const points = values.map((v,i)=>{
      const x = padding + (i/(Math.max(1, values.length-1)))*(w-2*padding)
      const y = h - padding - ((v - min)/range)*(h-2*padding)
      return [x,y]
    })
    // polyline
    const poly = document.createElementNS(svgNS, 'polyline')
    poly.setAttribute('fill', 'none')
    poly.setAttribute('stroke', 'rgba(110,231,183,0.9)')
    poly.setAttribute('stroke-width', '2')
    poly.setAttribute('points', points.map(p=>p.join(',')).join(' '))
    svg.appendChild(poly)
    // dots
    points.forEach(p=>{
      const c = document.createElementNS(svgNS, 'circle')
      c.setAttribute('cx', p[0])
      c.setAttribute('cy', p[1])
      c.setAttribute('r', 3)
      c.setAttribute('fill', 'rgba(110,231,183,0.9)')
      svg.appendChild(c)
    })
    wrap.appendChild(svg)
  }

  async function loadHistory(){
    try{
      const resp = await fetch('/predictions')
      const data = await resp.json()
      // data is list of entries (oldest->newest)
      const labels = data.map(d=>d.ts_readable)
      const values = data.map(d=>Number(d.prediction))
      if(chart){
        chart.data.labels = labels
        chart.data.datasets[0].data = values
        chart.update()
      } else {
        renderFallbackSVG(labels, values)
      }

      // fill table (most recent first)
      tableBody.innerHTML = ''
      data.slice().reverse().forEach(d=>{
        const tr = document.createElement('tr')
        const timeTd = document.createElement('td')
        timeTd.textContent = d.ts_readable
        const siTd = document.createElement('td')
        siTd.textContent = `${d.input.store} / ${d.input.item}`
        const inTd = document.createElement('td')
        inTd.textContent = `price=${d.input.price}, promo=${d.input.promotion}`
        const pTd = document.createElement('td')
        pTd.textContent = Number(d.prediction).toFixed(3)
        tr.appendChild(timeTd)
        tr.appendChild(siTd)
        tr.appendChild(inTd)
        tr.appendChild(pTd)
        tableBody.appendChild(tr)
      })
    }catch(err){
      console.error('Failed to load history', err)
    }
  }

  async function loadEda(){
    const container = document.getElementById('edaImages')
    container.innerHTML = ''
    const candidates = ['sales_distribution.png','sales_over_time.png','sales_by_store.png']
    for(const f of candidates){
      try{
        const resp = await fetch(`/outputs/${f}`, {method:'HEAD'})
        if(resp.ok){
          const img = document.createElement('img')
          img.src = `/outputs/${f}`
          img.style.width = '100%'
          img.style.marginBottom = '12px'
          container.appendChild(img)
        }
      }catch(e){
        // ignore
      }
    }
    if(!container.hasChildNodes()){
      container.innerHTML = '<p>No EDA outputs found. Run training/EDA to generate images.</p>'
    }
  }

  // initial load and poll every 3s
  loadHistory(); loadEda();
  setInterval(loadHistory, 3000)
})
