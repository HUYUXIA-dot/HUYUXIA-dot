var roi = table
Map.centerObject(roi, 10);
Map.addLayer(roi, {color:"black"}, "roi");
var l8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_TOA')
          .filterDate('2019-01-01', '2019-12-31')
          .filterBounds(roi)
          .map(roiClip);

function roiClip(image){
  return image.clip(roi)
}
function addNDVI(image) {
  var ndvi = image.normalizedDifference(['B5', 'B4']).rename('NDVI');
  return image.addBands(ndvi);
}
var withNDVI = l8.map(addNDVI);

var viz = {min:-1, max:1, palette:'blue, white, green'};
Map.addLayer(withNDVI.select('NDVI'), viz, 'NDVI');

var ndviTransform = function(img){ 
  var ndvi = img.normalizedDifference(['B4', 'B3']) // calculate normalized dif between band 4 and band 3 (B4-B3/B4_B3)
                .multiply(1000) // scale results by 1000
                .select([0], ['NDVI']) // name the band
                .set('system:time_start', img.get('system:time_start'));
  return ndvi;
};
var geometry = ee.FeatureCollection('users');
Map.centerObject(geometry,6);
var dataset = ee.ImageCollection('MODIS/006/MOD17A3HGF').filter(ee.Filter.date('2014-01-01', '2014-12-31'));
print(dataset)
var npp = dataset.select('Npp');
var nppVis = {
  min: 0.0,
  max: 19000.0,
  palette: ['bbe029', '0a9501', '074b03'],
};
Map.addLayer(npp.mean().clip(geometry), nppVis, 'NPP');

p1_npp_rs <- ggplot(s_npp_rs, aes(Rs_annual, NPP_annual, color = group)) +
  scale_x_log10(label = comma) + scale_y_log10(label = comma) +
  annotation_logticks() +
  xlab(expression(R[S]~(g~C~m^-2~yr^-1))) +
  ylab(expression(NPP~(g~C~m^-2~yr^-1))) + 
  coord_cartesian(xlim = c(80, 3300), ylim = c(70, 2000))
p1_npp_rs_clr <- p1_npp_rs + scale_color_discrete("Year")
p1_npp_rs_clr2 <- p1_npp_rs + scale_color_brewer("Year")

p_inset <- ggplot(s_npp_rs, aes(NPP_annual / Rs_annual, color = yeargroup, fill = yeargroup)) + 
  geom_density(alpha = 0.5) + 
  xlab(expression(NPP:R[S])) + ylab("") +
  theme(axis.ticks.y = element_blank(), axis.text.y  = element_blank(),
        axis.text.x = element_text(size = 6), axis.title.x = element_text(size = 8))
p_inset_clr <- p_inset +  scale_fill_discrete(guide = FALSE) +
  scale_color_discrete(guide = FALSE)
p_inset_clr2 <- p_inset +  scale_fill_brewer(guide = FALSE) +
  scale_color_brewer(guide = FALSE)

p1_npp_rs_clr <- p1_npp_rs_clr + 
  annotation_custom(grob = ggplotGrob(p_inset_clr), xmin = log10(60), xmax = log10(800), ymin = log10(600), ymax = log10(2300)) +
  geom_point() + geom_smooth(method = "lm", se = FALSE)
p1_npp_rs_clr2 <- p1_npp_rs_clr2 + 
  annotation_custom(grob = ggplotGrob(p_inset_clr2), xmin = log10(60), xmax = log10(800), ymin = log10(600), ymax = log10(2300)) +
  geom_point() + geom_smooth(method = "lm", se = FALSE)

printlog("NOTE we are plotting this graph with one point cut off:")
printlog(s_npp_rs[which.min(s_npp_rs$Rs_annual), c("Rs_annual", "NPP_annual")])

print(p1_npp_rs_clr)
save_plot("Fig1-srdb-npp-rs-clr", ptype = ".png", width = 9, height = 8)
print(p1_npp_rs_clr2)
save_plot("Fig1-srdb-npp-rs-clr2", ptype = ".png", width = 9, height = 8)
save_plot("Fig1-srdb-npp-rs-clr2", width = 9, height = 8)

p_npp_rs_time <- ggplot(s_npp_rs, aes(Study_midyear, NPP_annual/Rs_annual, color = Biome)) +
  geom_point() + geom_smooth(method = "lm", aes(group = 1)) +
  xlab("Year") + ylab(expression(NPP:R[S]))
print(p_npp_rs_time)
save_plot("Fig1-npp_rs_time", width = 6, height = 4)

def simulate_biomass(npp, rs):
    return biomass
def simulate_soc(npp, rs):
    return soc
model = load(r'E:\project\pythonProject\TECOR\ALL\random_tecor_model.joblib')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)


mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("训练集评价指标:")
print("均方误差 (MSE):", mse_train)
print("均方根误差 (RMSE):", rmse_train)
print("平均绝对误差 (MAE):", mae_train)
print("拟合优度 (R-squared):", r2_train)

print("\n测试集评价指标:")
print("均方误差 (MSE):", mse_test)
print("均方根误差 (RMSE):", rmse_test)
print("平均绝对误差 (MAE):", mae_test)
print("拟合优度 (R-squared):", r2_test)

data_train = pd.DataFrame({
    'True': y_train,
    'Predicted': y_pred_train,
    'Data Set': 'Train'
})

data_test = pd.DataFrame({
    'True': y_test,
    'Predicted': y_pred_test,
    'Data Set': 'Test'
})

data = pd.concat([data_train, data_test])
palette = {'Train': '#b4d4e1', 'Test': '#f4ba8a'}

plt.figure(figsize=(8, 6), dpi=1200)
g = sns.JointGrid(data=data, x="True", y="Predicted", hue="Data Set", height=10, palette=palette)
g.plot_joint(sns.scatterplot, alpha=0.5)
sns.regplot(data=data_train, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color='#b4d4e1', label='Train Regression Line')
sns.regplot(data=data_test, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color='#f4ba8a', label='Test Regression Line')
g.plot_marginals(sns.histplot, kde=False, element='bars', multiple='stack', alpha=0.5)
ax = g.ax_joint
ax.text(0.95, 0.1, f'Train $R^2$ = {r2_train:.3f}', transform=ax.transAxes, fontsize=12,
        verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
ax.text(0.95, 0.05, f'Test $R^2$ = {r2_test:.3f}', transform=ax.transAxes, fontsize=12,
        verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
ax.text(0.75, 0.99, 'Model = Random Forest', transform=ax.transAxes, fontsize=12,
        verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))
ax.plot([data['True'].min(), data['True'].max()], [data['True'].min(), data['True'].max()], c="black", alpha=0.5, linestyle='--', label='x=y')
ax.legend()
plt.savefig("TrueFalse.pdf", format='pdf', bbox_inches='tight')
plt.show()
