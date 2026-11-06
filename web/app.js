const form = document.querySelector('#analysis-form');
const query = document.querySelector('#query');
const result = document.querySelector('#result');
const submit = document.querySelector('#submit');
const state = document.querySelector('#result-state');
const labels = {classify:'Classifier le texte', search:'Rechercher les passages', ask:'Générer une synthèse'};
let activeMode = 'classify';
function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function sourceList(sources) {
  sources.forEach((source) => {
    const details = node('details', undefined, 'source');
    details.append(node('summary', `Réclamation ${source.complaint_id} · ${source.score?.toFixed(3) ?? 'source'}`));
    details.append(node('p', source.excerpt));
    details.append(node('small', [source.product, source.date_received, source.source_file].filter(Boolean).join(' · ')));
    result.append(details);
  });
}
query.addEventListener('input', () => {
  document.querySelector('#characters').textContent = `${query.value.length.toLocaleString('fr-FR')} / 10 000`;
});
document.querySelectorAll('[name="mode"]').forEach((radio) => radio.addEventListener('change', () => {
  activeMode = radio.value;
  document.querySelector('#submit-label').textContent = labels[activeMode];
  document.querySelector('.k-control').hidden = activeMode === 'classify';
}));
document.querySelectorAll('[data-example]').forEach((button) => button.addEventListener('click', () => {
  query.value = button.dataset.example;
  query.dispatchEvent(new Event('input'));
  query.focus();
}));
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!query.value.trim()) { query.focus(); query.setCustomValidity('Saisissez un texte non vide.'); query.reportValidity(); return; }
  query.setCustomValidity('');
  const mode = activeMode;
  submit.disabled = true;
  state.textContent = 'En cours';
  result.replaceChildren(node('div', mode === 'ask' ? 'Génération locale en cours…' : 'Analyse en cours…', 'loading'));
  try {
    const response = await fetch(`/${mode}`, {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text:query.value.trim(), k:Number(document.querySelector('#k').value)})});
    if (!response.ok) throw new Error(response.status === 422 ? 'Le texte ou les paramètres ne sont pas valides.' : 'Le service est momentanément indisponible. Réessayez ou consultez les journaux du serveur.');
    const data = await response.json();
    result.replaceChildren();
    if (mode === 'classify') {
      const prediction = node('div', undefined, 'prediction');
      prediction.append(node('span','PRODUIT FINANCIER IDENTIFIÉ','result-kicker'), node('h3',data.product),
        node('p','Modèle : TF-IDF + régression logistique.'), node('p','Catégorie prédite, sans estimation de confiance calibrée.'));
      result.append(prediction);
    } else if (mode === 'search') {
      result.append(node('h3',`${data.sources.length} passage(s) retrouvé(s)`));
      if (!data.sources.length) result.append(node('p','Aucun passage disponible pour cette recherche.'));
      sourceList(data.sources);
    } else {
      const answer = node('div',undefined,'answer');
      answer.append(node('h3', data.mode.startsWith('abstention') ? 'Preuves insuffisantes' : 'Réponse générée'));
      answer.append(node('div',data.answer,'answer-text'));
      if (data.warning) answer.append(node('div','Les références sont contrôlées. Vérifiez chaque affirmation dans les extraits.','warning'));
      result.append(answer);
      sourceList(data.sources);
    }
    state.textContent = 'Terminé';
  } catch (error) {
    result.replaceChildren(node('p',error.message,'error'));
    state.textContent = 'Erreur';
  } finally { submit.disabled = false; }
});
query.addEventListener('input', () => query.setCustomValidity(''));
async function loadProject() {
  try {
    const response = await fetch('/project');
    if (!response.ok) throw new Error('metadata');
    const data = await response.json();
    document.querySelector('#corpus-count').textContent = data.corpus.kept.toLocaleString('fr-FR');
    document.querySelector('#f1').textContent = data.temporal.macro_f1.toFixed(3).replace('.',',');
    document.querySelector('#test-count').textContent = `${data.temporal.rows_evaluated.toLocaleString('fr-FR')} récits · test 2026`;
    document.querySelector('#tests').textContent = data.tests;
    document.querySelector('#coverage').textContent = `${Math.round(data.coverage)} %`;
    document.querySelector('#connection').textContent = data.model_ready ? '● Service local actif' : '● Modèles à préparer';
  } catch {
    document.querySelector('#connection').textContent = 'Données de suivi indisponibles';
  }
}
loadProject();
